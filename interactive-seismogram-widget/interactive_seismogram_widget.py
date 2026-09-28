import warnings
warnings.filterwarnings("ignore") # no need to see these

import numpy as np
import pandas as pd
from obspy import read, Trace, Stream
from pathlib import Path
from bokeh.events import Tap
from bokeh.plotting import figure, show as _bokeh_show
from bokeh.layouts import column, row, gridplot
from bokeh.models import (
    ColumnDataSource,
    Span,
    Label,
    Div,
    Button,
    FullscreenTool,
    CrosshairTool,
    HoverTool,
    Select,
    Spinner,
    CheckboxGroup,
    ColorBar,
    LinearColorMapper,
)
import bokeh.palettes as palettes
from bokeh.io import output_notebook, reset_output
from urllib.parse import urlparse
from bokeh.io.notebook import HTML_MIME_TYPE, JS_MIME_TYPE, EXEC_MIME_TYPE
from bokeh.embed import server_document
from IPython.display import publish_display_data
from tornado.ioloop import IOLoop
from uuid import uuid4
import re
from bokeh.io.state import curstate
from bokeh.server.server import Server

import os
import asyncio

def _bokeh_notebook_url(port):
    """Tell Bokeh where the browser should reach the plot server."""
    
    service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")

    if service_prefix:
        return f"{service_prefix.rstrip('/')}/proxy/{port}"

    return f"http://localhost:{port}"


def use_notebook(local_url=None):
    """Prepare the notebook for plotting."""
    return output_notebook()


def use_standalone():
    return reset_output()


def _record_requesting_origin(handler):
    """Allow websockets from whichever page fetched the loader. """
    try:
        origins = handler.application.websocket_origins
    except AttributeError:
        return

    for header in ("Origin", "Referer"):
        value = handler.request.headers.get(header)

        if not value:
            continue

        host = urlparse(value).netloc.lower()

        if host:
            origins.add(host)
            return


def _display_bokeh_app(app):
    """Show a Bokeh server application in the notebook. """
    if curstate().notebook_type is None:
        server = Server({"/": app}, port=0)
        server.io_loop.add_callback(server.show, "/")

        try:
            asyncio.get_running_loop()
            loop_already_running = True
        except RuntimeError:
            loop_already_running = False

        if loop_already_running:
            server.start()
            print(
                f"Serving the picker at http://localhost:{server.port}/ "
                f"(running in the background of this kernel)"
            )
            return

        print(f"Serving the picker at http://localhost:{server.port}/ (Ctrl+C to stop)")
        server.run_until_shutdown()
        return

    loop = IOLoop.current()
    server = Server(
        {"/": app},
        io_loop=loop,
        port=0,
        log_function=_record_requesting_origin,
    )

    server_id = uuid4().hex
    curstate().uuid_to_server[server_id] = server
    server.start()

    script_tag = server_document(_bokeh_notebook_url(server.port), resources=None)

    match = re.search(r'<script id="([^"]+)">(.*)</script>', script_tag, re.DOTALL)
    if match is None:
        return _bokeh_show(app, notebook_url=_bokeh_notebook_url)

    elementid, loader_js = match.group(1), match.group(2)

    marker = "xhr.open('GET', " + chr(34)
    start = loader_js.find(marker)

    if start != -1:
        url_start = start + len(marker)
        url_end = loader_js.index(chr(34), url_start)
        path, _, query = loader_js[url_start:url_end].partition("?")
        loader_js = (
            loader_js[:url_start]
            + path
            + "?"
            + query.replace("%", "%25")
            + loader_js[url_end:]
        )

    publish_display_data({HTML_MIME_TYPE: f'<div id="{elementid}"></div>'})
    publish_display_data(
        {JS_MIME_TYPE: loader_js, EXEC_MIME_TYPE: ""},
        metadata={EXEC_MIME_TYPE: {"server_id": server_id}},
    )


def _normalize_selector(value, name):
    """Return a validated ObsPy selector or ``None``."""
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        raise ValueError(
            f"{name} must be a non-empty code or wildcard pattern."
        )

    return value


def _available_codes(stream, field):
    return sorted(
        {
            str(getattr(trace.stats, field, "") or "").strip()
            for trace in stream
            if str(getattr(trace.stats, field, "") or "").strip()
        }
    )


def _available_components(stream):
    return sorted(
        {
            channel[-1]
            for channel in _available_codes(stream, "channel")
            if channel
        }
    )


def _selector_kwargs(station=None, channel=None, component=None):
    """Build keyword arguments accepted by ``ObsPy Stream.select``."""
    station = _normalize_selector(station, "station")
    channel = _normalize_selector(channel, "channel")
    component = _normalize_selector(component, "component")

    selectors = {}
    if station is not None:
        selectors["station"] = station
    if channel is not None:
        selectors["channel"] = channel
    if component is not None:
        selectors["component"] = component

    return selectors


def _format_selectors(selectors):
    """Return selectors in a readable form for labels and errors."""
    return ", ".join(
        f"{name}={value!r}" for name, value in selectors.items()
    )


def _select_waveform_traces(
    stream,
    station=None,
    channel=None,
    component=None,
    source_name="waveform data",
):
    """Filter a Stream by station/channel/component with useful errors."""
    selectors = _selector_kwargs(
        station=station,
        channel=channel,
        component=component,
    )
    if not selectors:
        return stream

    selected = stream.select(**selectors)
    if len(selected) == 0:
        stations = _available_codes(stream, "station")
        channels = _available_codes(stream, "channel")
        components = _available_components(stream)
        raise ValueError(
            f"No traces matched {_format_selectors(selectors)} in {source_name}. "
            f"Available stations: {', '.join(stations) if stations else 'none reported'}. "
            f"Available channels: {', '.join(channels) if channels else 'none reported'}. "
            f"Available components: {', '.join(components) if components else 'none reported'}."
        )

    return selected


def _read_sac_directory(
    directory,
    station=None,
    channel=None,
    component=None,
):
    """Read SAC files, optionally limiting by station/channel/component."""
    directory = Path(directory).expanduser()
    selectors = _selector_kwargs(
        station=station,
        channel=channel,
        component=component,
    )

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory path: {directory}")

    sac_paths = []
    for pattern in ("*.sac", "*.SAC", "*.Sac"):
        sac_paths.extend(directory.glob(pattern))

    # Some SAC files have no useful extension, so fall back to trying every
    # regular file in the directory if extension matching found nothing.
    if not sac_paths:
        sac_paths = [path for path in directory.iterdir() if path.is_file()]

    sac_paths = sorted(set(sac_paths), key=lambda path: path.name.lower())

    if not sac_paths:
        raise ValueError(f"No files were found in directory: {directory}")

    stream = Stream()
    failures = []
    header_stream = Stream()

    for sac_path in sac_paths:
        try:
            if selectors:
                # SAC files normally contain one trace. Read only the header
                # first so nonmatching sample arrays are never loaded.
                file_header = read(str(sac_path), format="SAC", headonly=True)
                header_stream += file_header

                if len(file_header.select(**selectors)) == 0:
                    continue

            file_stream = read(str(sac_path), format="SAC")
            if selectors:
                file_stream = file_stream.select(**selectors)
            stream += file_stream

        except Exception as exc:
            failures.append((sac_path.name, str(exc)))

    if len(stream) == 0:
        if selectors:
            stations = _available_codes(header_stream, "station")
            channels = _available_codes(header_stream, "channel")
            components = _available_components(header_stream)
            raise ValueError(
                f"No SAC traces matched {_format_selectors(selectors)} in "
                f"directory: {directory}. "
                f"Available stations: {', '.join(stations) if stations else 'none reported'}. "
                f"Available channels: {', '.join(channels) if channels else 'none reported'}. "
                f"Available components: {', '.join(components) if components else 'none reported'}."
            )

        details = "; ".join(f"{name}: {error}" for name, error in failures[:5])
        raise ValueError(
            f"No SAC traces could be read from directory: {directory}. "
            f"First failures: {details}"
        )

    if failures:
        print(
            f"Warning: skipped {len(failures)} file(s) that could not be read as SAC. "
            f"First skipped: {failures[0][0]}"
        )

    return stream


def _coerce_waveform_data(
    data=None,
    station=None,
    channel=None,
    component=None,
):
    """Return a filtered ObsPy Stream plus a readable source label."""
    selectors = _selector_kwargs(
        station=station,
        channel=channel,
        component=component,
    )

    if data is None:
        stream = read()
        source_name = "ObsPy built-in example"

    elif isinstance(data, Stream):
        stream = data.copy()
        source_name = "ObsPy Stream"

    elif isinstance(data, Trace):
        stream = Stream(traces=[data.copy()])
        source_name = "ObsPy Trace"

    elif isinstance(data, (str, Path)):
        path = Path(data).expanduser()
        source_name = str(data)

        if path.is_dir():
            stream = _read_sac_directory(
                path,
                station=station,
                channel=channel,
                component=component,
            )
            source_name = f"SAC directory: {path}"
        else:
            stream = read(str(path))

    elif isinstance(data, (list, tuple)) and all(isinstance(item, Trace) for item in data):
        stream = Stream(traces=[item.copy() for item in data])
        source_name = f"{len(stream)} ObsPy traces"

    else:
        raise TypeError(
            "show(data) expects an ObsPy Stream, an ObsPy Trace, "
            "a SAC/miniSEED file path, a directory of SAC files, "
            "or a list/tuple of ObsPy Trace objects."
        )

    if len(stream) == 0:
        raise ValueError("No traces were found in the supplied waveform data.")

    # For miniSEED, individual SAC files, and already-created Streams, ObsPy
    # reads the source first and then applies all selectors here.
    stream = _select_waveform_traces(
        stream,
        station=station,
        channel=channel,
        component=component,
        source_name=source_name,
    )

    if selectors:
        source_name = f"{source_name} | {_format_selectors(selectors)}"

    return stream, source_name

def make_arrival_picker_app(
    doc,
    pick_state,
    data=None,
    trace_index=0,
    source_name=None,
    mode="single",
    station=None,
    channel=None,
    component=None,
):
    """Build either the single-trace picker or the component-stack picker."""
    if mode not in {"single", "components"}:
        raise ValueError("mode must be 'single' or 'components'.")

    stream, inferred_source_name = _coerce_waveform_data(
        data,
        station=station,
        channel=channel,
        component=component,
    )

    if source_name is None:
        source_name = inferred_source_name

    trace_index = int(trace_index)
    if trace_index < 0 or trace_index >= len(stream):
        raise IndexError(f"trace_index must be between 0 and {len(stream) - 1}.")

    if mode == "components" and len(stream) < 2:
        raise ValueError(
            "show_components() requires a Stream containing at least two traces."
        )

    def prepare_trace_arrays(trace, reftime=None):
        """Return (x, y) for a trace.

        With ``reftime`` set, x is seconds from that common instant, so
        traces that began at different times sit at their true offsets.
        Without it, x is seconds from the trace's own start, which aligns
        every trace on its own first sample.
        """
        if reftime is None:
            x_values = np.asarray(trace.times(), dtype=float).ravel()
        else:
            x_values = np.asarray(trace.times(reftime=reftime), dtype=float).ravel()

        y_values = np.asarray(trace.data, dtype=float).ravel()

        keep = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[keep]
        y_values = y_values[keep]

        if len(x_values) < 2:
            raise ValueError("Trace must contain at least two finite samples.")

        order = np.argsort(x_values, kind="mergesort")
        return x_values[order], y_values[order]

    def trace_id(trace):
        try:
            return trace.id
        except Exception:
            stats = trace.stats
            return ".".join(
                str(getattr(stats, item, "") or "")
                for item in ("network", "station", "location", "channel")
            ).strip(".")

    def trace_option_label(index, trace):
        stats = trace.stats
        return (
            f"{index}: {trace_id(trace)} | "
            f"{stats.starttime} | "
            f"{int(stats.npts)} samples | "
            f"{float(stats.sampling_rate):g} Hz"
        )

    input_stream = [one_trace.copy() for one_trace in stream]
    initial_trace = input_stream[trace_index].copy()
    initial_x, initial_y = prepare_trace_arrays(initial_trace)

    state = {
        "stream": input_stream,
        "trace": initial_trace,
        "x": initial_x,
        "raw_y": initial_y,
        "source_label": source_name,
    }
    state["x_min"] = float(initial_x.min())
    state["x_max"] = float(initial_x.max())
    state["fs"] = float(initial_trace.stats.sampling_rate)
    state["nyquist"] = state["fs"] / 2.0

    displayed_signal = {"values": initial_y.copy()}
    stacked_items = {"items": []}
    component_page = {"index": 0, "size": 15}

    source = ColumnDataSource(data={"x": initial_x, "y": initial_y})
    spec_source = ColumnDataSource(
        data={
            "image": [np.zeros((2, 2), dtype=float)],
            "x": [state["x_min"]],
            "y": [0.0],
            "dw": [float(state["x_max"] - state["x_min"])],
            "dh": [1.0],
        }
    )

    # pick_state belongs to the caller and holds nothing but picks, keyed by
    # trace id: {"IU.ANMO.00.BHZ": {"P": UTCDateTime, "S": UTCDateTime}}.
    # Times are absolute, so a pick means the same instant however the
    # traces happen to be plotted.
    pick_state.clear()

    # Everything the widget needs for itself is kept separately, so the
    # caller never has to pick trace ids out from among bookkeeping keys.
    # "active" maps a trace id to the phase its next pick will set. In
    # shared mode every displayed trace is picked together, so the
    # sequence is one thing and lives under the key None.
    widget_state = {"active": {}, "processing": "raw"}

    PHASES = ("P", "S")

    y_min = float(initial_y.min())
    y_max = float(initial_y.max())

    p = figure(
        width=1000,
        height=420,
        title=f"{trace_id(initial_trace)}",
        x_axis_label="Time (s)",
        y_axis_label="Amplitude",
        x_range=(state["x_min"], state["x_max"]),
        y_range=(y_min, y_max),
        tools="pan,wheel_zoom,box_zoom,tap,reset,save",
        active_scroll="wheel_zoom",
    )

    line_renderer = p.line(
        "x",
        "y",
        source=source,
        line_width=1,
        line_color="black",
    )

    spectrogram_palettes = {
        name: list(palette_sizes[256])
        for name, palette_sizes in palettes.all_palettes.items()
        if 256 in palette_sizes
    }
    default_color_scheme = (
        "Viridis"
        if "Viridis" in spectrogram_palettes
        else sorted(spectrogram_palettes)[0]
    )
    selected_color_scheme = spectrogram_palettes[default_color_scheme]

    spec_color_mapper = LinearColorMapper(
        palette=selected_color_scheme,
        low=-120.0,
        high=0.0,
    )

    spec_renderer = p.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=spec_source,
        color_mapper=spec_color_mapper,
        visible=False,
    )

    spec_color_bar = ColorBar(
        color_mapper=spec_color_mapper,
        label_standoff=8,
        title="dB",
        visible=False,
    )
    p.add_layout(spec_color_bar, "right")

    hover = HoverTool(
        renderers=[line_renderer],
        tooltips=[
            ("time", "@x{0.000000} s"),
            ("amp", "@y"),
        ],
        mode="vline",
        line_policy="nearest",
    )
    p.add_tools(hover)

    crosshair_v = Span(
        location=state["x_min"],
        dimension="height",
        line_color="gray",
        line_alpha=0.4,
        line_dash="dashed",
        line_width=1,
    )

    crosshair_h = Span(
        location=y_min,
        dimension="width",
        line_color="gray",
        line_alpha=0.4,
        line_dash="dashed",
        line_width=1,
    )

    p.add_tools(
        CrosshairTool(
            overlay=(crosshair_v, crosshair_h),
            dimensions="both",
        )
    )

    p.add_tools(FullscreenTool())

    p_span = Span(
        location=state["x_min"],
        dimension="height",
        line_color="red",
        line_width=2,
        line_dash="dashed",
        visible=False,
    )

    s_span = Span(
        location=state["x_min"],
        dimension="height",
        line_color="blue",
        line_width=2,
        line_dash="dashed",
        visible=False,
    )

    p.add_layout(p_span)
    p.add_layout(s_span)

    p_label = Label(
        x=state["x_min"],
        y=y_max,
        text="P",
        text_color="red",
        visible=False,
        text_baseline="top", 
        y_offset=-50,
        text_font_size = "32px",
    )

    s_label = Label(
        x=state["x_min"],
        y=y_max,
        text="S",
        text_color="blue",
        visible=False,
        text_baseline="top", 
        y_offset=-50,
        text_font_size = "32px",
    )

    status_label = Label(
        x=state["x_min"],
        y=y_max,
        text="Next pick: P | Data: raw",
        text_color="black",
        background_fill_color="white",
        background_fill_alpha=0.75,
        text_font_size="10pt",
        x_offset=10,
        y_offset=-25,
    )

    p.add_layout(p_label)
    p.add_layout(s_label)
    p.add_layout(status_label)

    # Keep a zero-size placeholder in the stacked panel so Bokeh does not
    # validate an empty layout before stacked-component plots are built.
    stacked_panel_placeholder = Div(text="", width=0, height=0)
    stacked_panel = column(stacked_panel_placeholder)
    stacked_panel.visible = False

    status = Div(text="<b>P:</b> — &nbsp;&nbsp; <b>S:</b> —")

    processing_status = Div(
        text=(
            f"<b>Processing:</b> raw &nbsp;&nbsp; "
            f"<b>Sampling rate:</b> {state['fs']:g} Hz &nbsp;&nbsp; "
            f"<b>Nyquist:</b> {state['nyquist']:g} Hz"
        )
    )

    input_status = Div(
        text=f"<b>Input:</b> {state['source_label']} &nbsp;&nbsp; <b>Trace:</b> {trace_id(initial_trace)}"
    )

    export_status = Div(text="<b>Export:</b> No file written yet.")

    view_status = Div(
        text="<b>View:</b> waveform. Use the View selector to switch to FFT, PSD, or spectrogram."
    )

    trace_select = None
    if mode == "single":
        trace_select = Select(
            title="Trace",
            value=str(trace_index),
            options=[
                (str(index), trace_option_label(index, one_trace))
                for index, one_trace in enumerate(state["stream"])
            ],
            width=700,
            disabled=len(state["stream"]) <= 1,
        )

    clear = Button(label="Clear picks")
    export_picks_button = Button(label="Export picks", button_type="success")
    apply_processing_button = Button(label="Apply processing", button_type="primary")
    reset_processing_button = Button(label="Reset raw")
    # Each pagination row needs its own Bokeh models, but both rows are
    # created by the same factory and share the same page state/callbacks.
    component_pagination_controls = []

    time_base_select = None
    if mode == "components":
        time_base_select = Select(
            title="Time base",
            value="absolute",
            options=[
                ("absolute", "Absolute"),
                ("relative", "Relative"),
            ],
            width=280,
        )

    view_select = Select(
        title="View",
        value="waveform",
        options=[
            ("waveform", "Waveform"),
            ("fft", "Fourier amplitude"),
            ("psd", "Power spectral density"),
            ("spectrogram", "Spectrogram"),
        ],
        width=180,
    )

    color_scheme_select = Select(
        title="Color scheme",
        value=default_color_scheme,
        options=sorted(spectrogram_palettes),
        width=150,
    )

    analysis_fmax_input = Spinner(
        title="max freq (Hz)",
        value=min(20.0, state["nyquist"]),
        low=0.1,
        high=max(state["nyquist"], 0.1),
        step=0.5,
        width=140,
    )

    spectrogram_window_input = Spinner(
        title="spec window (s)",
        value=2.0,
        low=max(1.0 / state["fs"], 0.01),
        high=max(float(state["x_max"] - state["x_min"]), 1.0 / state["fs"]),
        step=0.25,
        width=140,
    )

    spectrogram_overlap_input = Spinner(
        title="spec overlap (%)",
        value=75.0,
        low=0.0,
        high=95.0,
        step=5.0,
        width=150,
    )

    filter_select = Select(
        title="Filter",
        value="none",
        options=[
            ("none", "None"),
            ("bandpass", "Bandpass"),
            ("highpass", "Highpass"),
            ("lowpass", "Lowpass"),
        ],
        width=140,
    )

    freqmin_input = Spinner(
        title="fmin (Hz)",
        value=1.0,
        low=0.0,
        high=max(state["nyquist"] * 0.999, 0.001),
        step=0.1,
        width=120,
    )

    freqmax_input = Spinner(
        title="fmax (Hz)",
        value=min(10.0, state["nyquist"] * 0.8),
        low=0.0,
        high=max(state["nyquist"] * 0.999, 0.001),
        step=0.1,
        width=120,
    )

    corners_input = Spinner(
        title="corners",
        value=4,
        low=1,
        high=10,
        step=1,
        width=100,
    )

    taper_input = Spinner(
        title="taper %",
        value=5.0,
        low=0.0,
        high=50.0,
        step=1.0,
        width=100,
    )

    smooth_input = Spinner(
        title="smooth window samples",
        value=1,
        low=1,
        high=1001,
        step=2,
        width=160,
    )

    processing_options = CheckboxGroup(
        labels=[
            "demean",
            "linear detrend",
            "taper",
            "smooth",
        ],
        active=[],
        width=500,
    )

    def fmt(value):
        if value is None:
            return "—"
        return f"{value:.6f} s"

    def time_axis_label(reftime=None):
        if reftime is None:
            return "Time (s)"
        return f"Time (s) from {reftime}"

    def set_range(x_start, x_end, y_start, y_end):
        if not np.isfinite(x_start) or not np.isfinite(x_end) or x_start == x_end:
            x_start = 0.0
            x_end = 1.0

        if not np.isfinite(y_start) or not np.isfinite(y_end) or y_start == y_end:
            y_start = -1.0
            y_end = 1.0

        p.x_range.start = float(x_start)
        p.x_range.end = float(x_end)
        p.y_range.start = float(y_start)
        p.y_range.end = float(y_end)

    def waveform_y_range(values):
        values = np.asarray(values, dtype=float)
        ymin = float(np.nanmin(values))
        ymax = float(np.nanmax(values))

        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0

        pad = 0.05 * (ymax - ymin)
        return ymin - pad, ymax + pad

    def stacked_is_active():
        return (
            mode == "components"
            and view_select.value == "waveform"
            and len(state["stream"]) > 1
        )

    def set_display_visibility():
        use_stacked = stacked_is_active()
        p.visible = not use_stacked
        stacked_panel.visible = use_stacked

    def absolute_time_base():
        """True when stacked panels are drawn on a shared absolute axis."""
        return (
            mode == "components"
            and time_base_select is not None
            and time_base_select.value == "absolute"
        )

    def display_reftime():
        """The instant x=0 refers to, or None when each trace uses its own."""
        if not absolute_time_base():
            return None
        return min(one_trace.stats.starttime for one_trace in state["stream"])

    def pick_for(trace, phase):
        """Stored absolute pick time for a trace, or None."""
        return pick_state.get(trace_id(trace), {}).get(phase)

    def pick_x(trace, phase):
        """Where a stored pick falls on the axis this trace is drawn against."""
        picked = pick_for(trace, phase)
        if picked is None:
            return None

        reftime = display_reftime()
        origin = trace.stats.starttime if reftime is None else reftime
        return float(picked - origin)

    def store_pick(trace, phase, x_value):
        """Record a pick made at ``x_value`` on this trace's axis."""
        reftime = display_reftime()
        origin = trace.stats.starttime if reftime is None else reftime
        pick_state.setdefault(trace_id(trace), {})[phase] = origin + float(x_value)

    def picks_are_shared():
        """True when one tap marks the same arrival on every panel."""
        return stacked_is_active() and not absolute_time_base()

    def phase_key(trace):
        return None if picks_are_shared() else trace_id(trace)

    def next_phase(trace):
        """Which phase the next tap on this trace will set."""
        return widget_state["active"].setdefault(phase_key(trace), "P")

    def advance_phase(trace):
        key = phase_key(trace)
        widget_state["active"][key] = "S" if widget_state["active"].get(key) == "P" else "P"

    def set_pick_marker_locations():
        current_trace = state["trace"]

        for phase, span, label in (
            ("P", p_span, p_label),
            ("S", s_span, s_label),
        ):
            x_value = pick_x(current_trace, phase)
            if x_value is not None:
                span.location = x_value
                label.x = x_value

        for item in stacked_items["items"]:
            for phase, span_key, label_key in (
                ("P", "p_span", "p_label"),
                ("S", "s_span", "s_label"),
            ):
                x_value = pick_x(item["trace"], phase)
                if x_value is not None:
                    item[span_key].location = x_value
                    item[label_key].x = x_value

    def show_pick_overlays(show=True):
        waveform_view = show and view_select.value == "waveform"
        use_stacked = stacked_is_active()
        show_single = waveform_view and not use_stacked
        show_stacked = waveform_view and use_stacked
        current_trace = state["trace"]

        has_p = pick_for(current_trace, "P") is not None
        has_s = pick_for(current_trace, "S") is not None

        p_span.visible = show_single and has_p
        s_span.visible = show_single and has_s
        p_label.visible = show_single and has_p
        s_label.visible = show_single and has_s
        status_label.visible = show_single

        if show_single:
            ymax = p.y_range.end
            p_label.y = ymax
            s_label.y = ymax
            status_label.y = ymax
            status_label.x = p.x_range.start

        for item in stacked_items["items"]:
            plot = item["plot"]
            ymax = plot.y_range.end
            item_p = pick_for(item["trace"], "P") is not None
            item_s = pick_for(item["trace"], "S") is not None

            item["p_span"].visible = show_stacked and item_p
            item["s_span"].visible = show_stacked and item_s
            item["p_label"].visible = show_stacked and item_p
            item["s_label"].visible = show_stacked and item_s
            item["p_label"].y = ymax
            item["s_label"].y = ymax

        set_pick_marker_locations()

    def refresh_status():
        processing = widget_state["processing"]

        if stacked_is_active() and stacked_items["items"]:
            shared = picks_are_shared()
            lines = []

            for item in stacked_items["items"]:
                one_trace = item["trace"]

                if pick_for(one_trace, "P") is None and pick_for(one_trace, "S") is None:
                    continue

                line = (
                    f"{trace_id(one_trace)} &nbsp; "
                    f"<b>P:</b> {fmt(pick_x(one_trace, 'P'))} &nbsp; "
                    f"<b>S:</b> {fmt(pick_x(one_trace, 'S'))}"
                )

                # With picks made per trace, each one is at its own point
                # in the sequence, so saying which phase comes next is
                # only meaningful alongside that trace.
                if not shared:
                    line += f" &nbsp; <i>next: {next_phase(one_trace)}</i>"

                lines.append(line)

            if shared:
                heading = f"<b>Next pick:</b> {widget_state['active'].get(None, 'P')}"
            else:
                heading = "<b>Picking each trace separately</b>"

            if lines:
                status.text = heading + "<br>" + "<br>".join(lines)
            else:
                status.text = heading + " &nbsp;&nbsp; No picks yet."
        else:
            current_trace = state["trace"]
            status.text = (
                f"<b>Next pick:</b> {next_phase(current_trace)} &nbsp;&nbsp; "
                f"<b>P:</b> {fmt(pick_x(current_trace, 'P'))} &nbsp;&nbsp; "
                f"<b>S:</b> {fmt(pick_x(current_trace, 'S'))}"
            )

        status_label.text = (
            f"Next pick: {next_phase(state['trace'])} | "
            f"P: {fmt(pick_x(state['trace'], 'P'))} | "
            f"S: {fmt(pick_x(state['trace'], 'S'))} | "
            f"Data: {processing}"
        )

        show_pick_overlays(True)

    def moving_average(values, window):
        window = int(window)

        if window <= 1:
            return values

        if window % 2 == 0:
            window += 1

        kernel = np.ones(window, dtype=float) / window
        return np.convolve(values, kernel, mode="same")

    def finite_demeaned(values):
        values = np.asarray(values, dtype=float)

        if not np.all(np.isfinite(values)):
            finite = np.isfinite(values)
            replacement = float(np.nanmean(values[finite])) if finite.any() else 0.0
            values = np.where(finite, values, replacement)

        return values - float(np.mean(values))

    def amplitude_spectrum(values):
        values = finite_demeaned(values)
        n = len(values)

        if n < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        fs = state["fs"]
        window = np.hanning(n)
        coherent_gain = float(np.mean(window))

        if coherent_gain == 0:
            coherent_gain = 1.0

        spectrum = np.fft.rfft(values * window)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        amp = 2.0 * np.abs(spectrum) / (n * coherent_gain)

        if len(amp):
            amp[0] = amp[0] / 2.0

        return freqs, amp

    def periodogram_psd_db(values):
        values = finite_demeaned(values)
        n = len(values)

        if n < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        fs = state["fs"]
        window = np.hanning(n)
        scale = fs * float(np.sum(window ** 2))

        if scale == 0:
            scale = 1.0

        spectrum = np.fft.rfft(values * window)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = (np.abs(spectrum) ** 2) / scale

        if len(psd) > 2:
            psd[1:-1] *= 2.0

        psd_db = 10.0 * np.log10(psd + np.finfo(float).tiny)
        return freqs, psd_db

    def spectrogram_db(values):
        values = finite_demeaned(values)
        n = len(values)
        fs = state["fs"]
        x_values = state["x"]

        if n < 2:
            return np.array([0.0]), np.array([state["x_min"]]), np.zeros((1, 1), dtype=float)

        nperseg = int(round(float(spectrogram_window_input.value) * fs))
        nperseg = max(8, min(nperseg, n))

        overlap_fraction = float(spectrogram_overlap_input.value) / 100.0
        overlap_fraction = max(0.0, min(overlap_fraction, 0.95))
        step = max(1, int(round(nperseg * (1.0 - overlap_fraction))))

        starts = list(range(0, n - nperseg + 1, step))
        if not starts:
            starts = [0]
        if starts[-1] != n - nperseg:
            starts.append(n - nperseg)

        window = np.hanning(nperseg)
        scale = fs * float(np.sum(window ** 2))
        if scale == 0:
            scale = 1.0

        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        columns = []
        times = []

        for start in starts:
            segment = values[start:start + nperseg]
            spectrum = np.fft.rfft(segment * window)
            psd = (np.abs(spectrum) ** 2) / scale

            if len(psd) > 2:
                psd[1:-1] *= 2.0

            columns.append(10.0 * np.log10(psd + np.finfo(float).tiny))
            times.append(float(x_values[start] + (nperseg / (2.0 * fs))))

        sxx_db = np.asarray(columns, dtype=float).T
        return freqs, np.asarray(times, dtype=float), sxx_db

    def visible_frequency_limit():
        fmax = float(analysis_fmax_input.value)
        return max(0.1, min(fmax, state["nyquist"]))

    def make_processing_label():
        parts = []

        if 0 in processing_options.active:
            parts.append("demean")

        if 1 in processing_options.active:
            parts.append("linear detrend")

        if 2 in processing_options.active:
            parts.append(f"taper {float(taper_input.value):g}%")

        filter_kind = filter_select.value

        if filter_kind == "bandpass":
            parts.append(
                f"bandpass {float(freqmin_input.value):g}–{float(freqmax_input.value):g} Hz"
            )
        elif filter_kind == "highpass":
            parts.append(f"highpass {float(freqmin_input.value):g} Hz")
        elif filter_kind == "lowpass":
            parts.append(f"lowpass {float(freqmax_input.value):g} Hz")

        if 3 in processing_options.active:
            parts.append(f"smooth {int(smooth_input.value)} samples")

        if not parts:
            return "raw"

        return " + ".join(parts)

    def process_trace_values(trace):
        y_proc = np.asarray(trace.data, dtype=float).copy()
        fs = float(trace.stats.sampling_rate)
        nyquist = fs / 2.0
        header = trace.stats.copy()
        proc_tr = Trace(data=y_proc, header=header)

        if 0 in processing_options.active:
            proc_tr.detrend("demean")

        if 1 in processing_options.active:
            proc_tr.detrend("linear")

        if 2 in processing_options.active:
            taper_fraction = float(taper_input.value) / 100.0
            if taper_fraction > 0:
                proc_tr.taper(max_percentage=taper_fraction, type="cosine")

        filter_kind = filter_select.value
        fmin = float(freqmin_input.value)
        fmax = float(freqmax_input.value)
        corners = int(corners_input.value)

        if filter_kind == "bandpass":
            if not (0 < fmin < fmax < nyquist):
                raise ValueError(
                    f"Bandpass requires 0 < fmin < fmax < Nyquist for every displayed trace. "
                    f"{trace_id(trace)} Nyquist = {nyquist:g} Hz."
                )

            proc_tr.filter(
                "bandpass",
                freqmin=fmin,
                freqmax=fmax,
                corners=corners,
            )

        elif filter_kind == "highpass":
            if not (0 < fmin < nyquist):
                raise ValueError(
                    f"Highpass requires 0 < fmin < Nyquist for every displayed trace. "
                    f"{trace_id(trace)} Nyquist = {nyquist:g} Hz."
                )

            proc_tr.filter(
                "highpass",
                freq=fmin,
                corners=corners,
            )

        elif filter_kind == "lowpass":
            if not (0 < fmax < nyquist):
                raise ValueError(
                    f"Lowpass requires 0 < fmax < Nyquist for every displayed trace. "
                    f"{trace_id(trace)} Nyquist = {nyquist:g} Hz."
                )

            proc_tr.filter(
                "lowpass",
                freq=fmax,
                corners=corners,
            )

        y_proc = np.asarray(proc_tr.data, dtype=float)

        if 3 in processing_options.active:
            y_proc = moving_average(y_proc, int(smooth_input.value))

        return y_proc

    def process_signal():
        return process_trace_values(state["trace"])

    def component_page_bounds():
        total = len(state["stream"])
        page_size = component_page["size"]
        page_count = max(1, (total + page_size - 1) // page_size)
        component_page["index"] = min(max(component_page["index"], 0), page_count - 1)
        start = component_page["index"] * page_size
        end = min(start + page_size, total)
        return start, end, page_count

    def refresh_component_pagination():
        start, end, page_count = component_page_bounds()
        total = len(state["stream"])
        current_page = component_page["index"] + 1

        status_text = (
            f"<b>Traces:</b> {start + 1}–{end} of {total} "
            f"&nbsp;&nbsp; <b>Page:</b> {current_page} of {page_count}"
        )

        for controls in component_pagination_controls:
            controls["previous"].disabled = component_page["index"] == 0
            controls["next"].disabled = component_page["index"] >= page_count - 1
            controls["previous"].visible = page_count > 1
            controls["next"].visible = page_count > 1
            controls["status"].text = status_text

    def build_stacked_waveform_view():
        figures = []
        items = []
        start, end, _ = component_page_bounds()
        stream_traces = state["stream"][start:end]
        shared_x_range = p.x_range

        x_mins = []
        x_maxs = []

        reftime = display_reftime()

        for index, trace in enumerate(stream_traces):
            x_values, _ = prepare_trace_arrays(trace, reftime=reftime)
            y_values = process_trace_values(trace)

            x_mins.append(float(np.nanmin(x_values)))
            x_maxs.append(float(np.nanmax(x_values)))

            ymin, ymax = waveform_y_range(y_values)
            height = 180 if len(stream_traces) <= 4 else 140
            tools = "pan,wheel_zoom,box_zoom,tap,reset,save"

            fig = figure(
                width=1000,
                frame_height=height,
                title=trace_id(trace),
                x_axis_label=(
                    time_axis_label(reftime)
                    if index == len(stream_traces) - 1
                    else ""
                ),
                y_axis_label=str(getattr(trace.stats, "channel", "") or "Amplitude"),
                x_range=shared_x_range,
                y_range=(ymin, ymax),
                tools=tools,
                active_scroll="wheel_zoom",
            )

            component_source = ColumnDataSource(data={"x": x_values, "y": y_values})
            component_renderer = fig.line(
                "x",
                "y",
                source=component_source,
                line_width=1,
                line_color="black",
            )

            component_hover = HoverTool(
                renderers=[component_renderer],
                tooltips=[
                    ("trace", trace_id(trace)),
                    ("time", "@x{0.000000} s"),
                    ("amp", "@y"),
                ],
                mode="vline",
                line_policy="nearest",
            )
            fig.add_tools(component_hover)

            component_crosshair_v = Span(
                location=float(x_values[0]),
                dimension="height",
                line_color="gray",
                line_alpha=0.4,
                line_dash="dashed",
                line_width=1,
            )
            component_crosshair_h = Span(
                location=ymin,
                dimension="width",
                line_color="gray",
                line_alpha=0.4,
                line_dash="dashed",
                line_width=1,
            )
            fig.add_tools(
                CrosshairTool(
                    overlay=(component_crosshair_v, component_crosshair_h),
                    dimensions="both",
                )
            )
            if index != len(stream_traces) - 1:
                fig.xaxis.visible = False

            component_p_span = Span(
                location=float(x_values[0]),
                dimension="height",
                line_color="red",
                line_width=2,
                line_dash="dashed",
                visible=False,
            )
            component_s_span = Span(
                location=float(x_values[0]),
                dimension="height",
                line_color="blue",
                line_width=2,
                line_dash="dashed",
                visible=False,
            )
            fig.add_layout(component_p_span)
            fig.add_layout(component_s_span)

            component_p_label = Label(
                x=float(x_values[0]),
                y=ymax,
                text="P",
                text_color="red",
                visible=False,
                text_baseline="top", 
                y_offset=-10,
                text_font_size = "24px",
            )
            component_s_label = Label(
                x=float(x_values[0]),
                y=ymax,
                text="S",
                text_color="blue",
                visible=False,
                text_baseline="top", 
                y_offset=-10,
                text_font_size = "24px",
            )
            fig.add_layout(component_p_label)
            fig.add_layout(component_s_label)

            fig.on_event(Tap, make_on_tap(trace, x_values))

            figures.append(fig)
            items.append(
                {
                    "plot": fig,
                    "trace": trace,
                    "source": component_source,
                    "p_span": component_p_span,
                    "s_span": component_s_span,
                    "p_label": component_p_label,
                    "s_label": component_s_label,
                }
            )

        if x_mins and x_maxs:
            p.x_range.start = min(x_mins)
            p.x_range.end = max(x_maxs)

        stacked_grid = gridplot(
            [[fig] for fig in figures],
            toolbar_location="right",
            merge_tools=True,
        )
        stacked_panel.children = [stacked_grid]
        stacked_items["items"] = items
        refresh_component_pagination()
        set_pick_marker_locations()
        show_pick_overlays(True)

    def update_view():
        values = np.asarray(displayed_signal["values"], dtype=float)
        view = view_select.value
        fmax = visible_frequency_limit()
        x_values = state["x"]
        x_min = state["x_min"]
        x_max = state["x_max"]
        fs = state["fs"]
        current_trace = state["trace"]

        try:
            set_display_visibility()

            if stacked_is_active():
                line_renderer.visible = False
                spec_renderer.visible = False
                spec_color_bar.visible = False
                hover.renderers = []
                build_stacked_waveform_view()

                if absolute_time_base():
                    view_status.text = (
                        "<b>View:</b> stacked components on a shared absolute "
                        "time base, so traces sit at their true offsets. "
                        "Clicking picks on that component only."
                    )
                else:
                    view_status.text = (
                        "<b>View:</b> stacked components, each aligned on its "
                        "own first sample. Clicking picks the same arrival on "
                        "every component shown."
                    )
                return

            if view == "waveform":
                line_renderer.visible = True
                spec_renderer.visible = False
                spec_color_bar.visible = False
                hover.renderers = [line_renderer]
                hover.tooltips = [
                    ("time", "@x{0.000000} s"),
                    ("amp", "@y"),
                ]
                source.data = {"x": x_values, "y": values}
                ymin, ymax = waveform_y_range(values)
                set_range(x_min, x_max, ymin, ymax)
                p.title.text = f"{trace_id(current_trace)}"
                p.xaxis.axis_label = "Time (s)"
                p.yaxis.axis_label = "Amplitude"
                view_status.text = "<b>View:</b> waveform. Click the trace to pick P/S arrivals."

            elif view == "fft":
                freqs, amp = amplitude_spectrum(values)
                keep_freq = freqs <= fmax
                freqs = freqs[keep_freq]
                amp = amp[keep_freq]
                source.data = {"x": freqs, "y": amp}
                line_renderer.visible = True
                spec_renderer.visible = False
                spec_color_bar.visible = False
                hover.renderers = [line_renderer]
                hover.tooltips = [
                    ("frequency", "@x{0.000000} Hz"),
                    ("amplitude", "@y"),
                ]
                ymax = float(np.nanmax(amp)) if len(amp) else 1.0
                set_range(0.0, fmax, 0.0, ymax * 1.05 if ymax > 0 else 1.0)
                p.title.text = f"{trace_id(current_trace)}"
                p.xaxis.axis_label = "Frequency (Hz)"
                p.yaxis.axis_label = "Amplitude"
                view_status.text = "<b>View:</b> Fourier amplitude spectrum of the displayed signal."

            elif view == "psd":
                freqs, psd_db = periodogram_psd_db(values)
                keep_freq = freqs <= fmax
                freqs = freqs[keep_freq]
                psd_db = psd_db[keep_freq]
                source.data = {"x": freqs, "y": psd_db}
                line_renderer.visible = True
                spec_renderer.visible = False
                spec_color_bar.visible = False
                hover.renderers = [line_renderer]
                hover.tooltips = [
                    ("frequency", "@x{0.000000} Hz"),
                    ("PSD", "@y{0.000} dB"),
                ]
                ymin = float(np.nanmin(psd_db)) if len(psd_db) else -1.0
                ymax = float(np.nanmax(psd_db)) if len(psd_db) else 1.0
                pad = 0.05 * (ymax - ymin) if ymax != ymin else 1.0
                set_range(0.0, fmax, ymin - pad, ymax + pad)
                p.title.text = f"{trace_id(current_trace)}"
                p.xaxis.axis_label = "Frequency (Hz)"
                p.yaxis.axis_label = "Power / Hz (dB)"
                view_status.text = "<b>View:</b> periodogram power spectral density of the displayed signal."

            elif view == "spectrogram":
                freqs, spec_times, sxx_db = spectrogram_db(values)
                keep_freq = freqs <= fmax
                freqs = freqs[keep_freq]
                sxx_db = sxx_db[keep_freq, :]

                if sxx_db.size == 0:
                    sxx_db = np.zeros((1, 1), dtype=float)
                    freqs = np.array([0.0])

                finite = sxx_db[np.isfinite(sxx_db)]
                if finite.size:
                    color_low = float(np.percentile(finite, 5))
                    color_high = float(np.percentile(finite, 98))
                else:
                    color_low = -1.0
                    color_high = 1.0

                if color_low == color_high:
                    color_low -= 1.0
                    color_high += 1.0

                spec_color_mapper.low = color_low
                spec_color_mapper.high = color_high

                spec_source.data = {
                    "image": [sxx_db],
                    "x": [x_min],
                    "y": [0.0],
                    "dw": [max(x_max - x_min, 1.0 / fs)],
                    "dh": [max(float(freqs[-1]) if len(freqs) else fmax, 1.0 / fs)],
                }

                line_renderer.visible = False
                spec_renderer.visible = True
                spec_color_bar.visible = True
                hover.renderers = []
                set_range(x_min, x_max, 0.0, fmax)
                p.title.text = f"{trace_id(current_trace)}"
                p.xaxis.axis_label = "Time (s)"
                p.yaxis.axis_label = "Frequency (Hz)"
                view_status.text = (
                    f"<b>View:</b> spectrogram of the displayed signal. "
                    f"Window = {float(spectrogram_window_input.value):g} s; "
                    f"overlap = {float(spectrogram_overlap_input.value):g}%; "
                    f"color scheme = {color_scheme_select.value}."
                )

            show_pick_overlays(True)

        except Exception as err:
            view_status.text = f"<b>View error:</b> {err}"

    def apply_processing():
        try:
            y_proc = process_signal()
            displayed_signal["values"] = y_proc

            processing_label = make_processing_label()
            widget_state["processing"] = processing_label

            processing_status.text = (
                f"<b>Processing:</b> {processing_label} &nbsp;&nbsp; "
                f"<b>Sampling rate:</b> {state['fs']:g} Hz &nbsp;&nbsp; "
                f"<b>Nyquist:</b> {state['nyquist']:g} Hz"
            )

            update_view()
            refresh_status()

        except Exception as err:
            processing_status.text = f"<b>Processing error:</b> {err}"

    def reset_processing():
        displayed_signal["values"] = state["raw_y"].copy()
        widget_state["processing"] = "raw"

        processing_status.text = (
            f"<b>Processing:</b> raw &nbsp;&nbsp; "
            f"<b>Sampling rate:</b> {state['fs']:g} Hz &nbsp;&nbsp; "
            f"<b>Nyquist:</b> {state['nyquist']:g} Hz"
        )

        update_view()
        refresh_status()

    def reset_picks():
        pick_state.clear()
        widget_state["active"].clear()

        p_span.visible = False
        s_span.visible = False
        p_label.visible = False
        s_label.visible = False

        for item in stacked_items["items"]:
            item["p_span"].visible = False
            item["s_span"].visible = False
            item["p_label"].visible = False
            item["s_label"].visible = False

    def update_control_limits_for_trace():
        fs = state["fs"]
        nyquist = state["nyquist"]
        duration = max(state["x_max"] - state["x_min"], 1.0 / fs)

        analysis_fmax_input.high = max(nyquist, 0.1)
        analysis_fmax_input.value = min(max(float(analysis_fmax_input.value), 0.1), analysis_fmax_input.high)

        if analysis_fmax_input.value > nyquist:
            analysis_fmax_input.value = min(20.0, nyquist)

        spectrogram_window_input.low = max(1.0 / fs, 0.01)
        spectrogram_window_input.high = max(duration, spectrogram_window_input.low)
        spectrogram_window_input.value = min(
            max(float(spectrogram_window_input.value), spectrogram_window_input.low),
            spectrogram_window_input.high,
        )

        freqmin_input.high = max(nyquist * 0.999, 0.001)
        freqmax_input.high = max(nyquist * 0.999, 0.001)
        freqmin_input.value = min(max(float(freqmin_input.value), 0.0), freqmin_input.high)
        freqmax_input.value = min(max(float(freqmax_input.value), 0.0), freqmax_input.high)

        if freqmax_input.value <= freqmin_input.value:
            freqmax_input.value = min(max(freqmin_input.value + 0.1, nyquist * 0.8), freqmax_input.high)

    def load_trace_into_app(new_trace, source_name=None):
        x_values, y_values = prepare_trace_arrays(new_trace)

        state["trace"] = new_trace.copy()
        state["x"] = x_values
        state["raw_y"] = y_values
        state["x_min"] = float(x_values.min())
        state["x_max"] = float(x_values.max())
        state["fs"] = float(new_trace.stats.sampling_rate)
        state["nyquist"] = state["fs"] / 2.0

        if source_name is not None:
            state["source_label"] = source_name

        displayed_signal["values"] = y_values.copy()
        widget_state["processing"] = "raw"
        filter_select.value = "none"
        processing_options.active = []
        update_control_limits_for_trace()

        processing_status.text = (
            f"<b>Processing:</b> raw &nbsp;&nbsp; "
            f"<b>Sampling rate:</b> {state['fs']:g} Hz &nbsp;&nbsp; "
            f"<b>Nyquist:</b> {state['nyquist']:g} Hz"
        )
        input_status.text = (
            f"<b>Input:</b> {state['source_label']} &nbsp;&nbsp; "
            f"<b>Trace:</b> {trace_id(state['trace'])}"
        )
        export_status.text = "<b>Export:</b> No file written yet."

        update_view()
        refresh_status()

    def on_trace_select_change(attr, old, new):
        try:
            index = int(new)
            selected_trace = state["stream"][index]
            load_trace_into_app(
                selected_trace,
                source_name=state["source_label"],
            )
        except Exception as err:
            input_status.text = f"<b>Trace selection error:</b> {err}"

    def nearest_time(clicked_x, x_values=None):
        if x_values is None:
            x_values = state["x"]

        i = np.searchsorted(x_values, clicked_x)

        if i <= 0:
            return float(x_values[0])

        if i >= len(x_values):
            return float(x_values[-1])

        left = x_values[i - 1]
        right = x_values[i]

        if abs(clicked_x - left) <= abs(clicked_x - right):
            return float(left)

        return float(right)

    def make_on_tap(tapped_trace=None, x_values=None):
        """Tap handler for one plot. """

        def handler(event):
            if view_select.value != "waveform":
                return

            source_trace = tapped_trace if tapped_trace is not None else state["trace"]
            picked_x = nearest_time(event.x, x_values)
            phase = next_phase(source_trace)

            if picks_are_shared():
                targets = [item["trace"] for item in stacked_items["items"]]
            else:
                targets = [source_trace]

            for one_trace in targets:
                store_pick(one_trace, phase, picked_x)

            advance_phase(source_trace)

            set_pick_marker_locations()
            refresh_status()

        return handler

    on_tap = make_on_tap()

    def clear_picks():
        reset_picks()
        refresh_status()

    def export_picks():
        """Write every pick to one file, one row per trace per phase."""
        rows = []

        by_id = {trace_id(one_trace): one_trace for one_trace in state["stream"]}

        for tid in sorted(pick_state):
            one_trace = by_id.get(tid)

            for phase in PHASES:
                picked_time = pick_state[tid].get(phase)

                if picked_time is None:
                    continue

                offset = (
                    float(picked_time - one_trace.stats.starttime)
                    if one_trace is not None
                    else float("nan")
                )

                rows.append(
                    {
                        "trace_id": tid,
                        "station": (
                            str(getattr(one_trace.stats, "station", "") or "unknown")
                            if one_trace is not None
                            else "unknown"
                        ),
                        "channel": (
                            str(getattr(one_trace.stats, "channel", "") or "unknown")
                            if one_trace is not None
                            else "unknown"
                        ),
                        "phase": phase,
                        "time_utc": str(picked_time),
                        "offset_s": f"{offset:.6f}",
                    }
                )

        if not rows:
            export_status.text = "<b>Export:</b> No picks to export."
            return

        out_dir = Path("picks")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "picks.txt"

        pd.DataFrame(rows).to_csv(
            out_path,
            sep="\t",
            index=False,
        )

        traces_with_picks = len({row["trace_id"] for row in rows})
        export_status.text = (
            f"<b>Exported:</b> {len(rows)} pick(s) across "
            f"{traces_with_picks} trace(s) to <code>{out_path}</code>"
        )

    def on_range_change(attr, old, new):
        if view_select.value == "waveform":
            status_label.x = p.x_range.start
            status_label.y = p.y_range.end
            for item in stacked_items["items"]:
                item["p_label"].y = item["plot"].y_range.end
                item["s_label"].y = item["plot"].y_range.end

    p.x_range.on_change("start", on_range_change)
    p.x_range.on_change("end", on_range_change)
    p.y_range.on_change("start", on_range_change)
    p.y_range.on_change("end", on_range_change)

    p.on_event(Tap, on_tap)

    clear.on_click(clear_picks)
    export_picks_button.on_click(export_picks)
    apply_processing_button.on_click(apply_processing)
    reset_processing_button.on_click(reset_processing)

    def show_previous_component_page():
        if component_page["index"] > 0:
            component_page["index"] -= 1
            update_view()

    def show_next_component_page():
        _, _, page_count = component_page_bounds()
        if component_page["index"] < page_count - 1:
            component_page["index"] += 1
            update_view()

    def make_component_pagination_controls():
        previous_button = Button(label="Previous 15", width=120)
        next_button = Button(label="Next 15", width=120)
        page_status = Div(text="", width=260)

        previous_button.on_click(show_previous_component_page)
        next_button.on_click(show_next_component_page)

        control_set = {
            "previous": previous_button,
            "next": next_button,
            "status": page_status,
        }
        component_pagination_controls.append(control_set)

        return row(previous_button, next_button, page_status)

    def update_analysis_control_visibility():
        view = view_select.value

        # Frequency limit applies to all frequency-domain views.
        analysis_fmax_input.visible = view in {
            "fft",
            "psd",
            "spectrogram",
        }

        # Window, overlap, and color scheme apply only to the spectrogram.
        spectrogram_window_input.visible = view == "spectrogram"
        spectrogram_overlap_input.visible = view == "spectrogram"
        color_scheme_select.visible = view == "spectrogram"

    def on_time_base_change(attr, old, new):
        # A pick made against one reference is meaningless against the
        # other, so start clean rather than silently reinterpreting it.
        reset_picks()
        update_view()
        refresh_status()

    def on_color_scheme_change(attr, old, new):
        spec_color_mapper.palette = spectrogram_palettes[new]
        update_view()

    def on_view_control_change(attr, old, new):
        update_analysis_control_visibility()
        update_view()

    if trace_select is not None:
        trace_select.on_change("value", on_trace_select_change)

    view_select.on_change("value", on_view_control_change)
    if time_base_select is not None:
        time_base_select.on_change("value", on_time_base_change)
    color_scheme_select.on_change("value", on_color_scheme_change)
    analysis_fmax_input.on_change("value", on_view_control_change)
    spectrogram_window_input.on_change("value", on_view_control_change)
    spectrogram_overlap_input.on_change("value", on_view_control_change)

    processing_controls = [
        row(filter_select, freqmin_input, freqmax_input, corners_input),
        row(taper_input, smooth_input),
        processing_options,
        row(apply_processing_button, reset_processing_button),
    ]

    if mode == "single":
        controls = column(
            trace_select,
            row(
                view_select,
                color_scheme_select,
                analysis_fmax_input,
                spectrogram_window_input,
                spectrogram_overlap_input,
            ),
            *processing_controls,
        )
    else:
        controls = column(time_base_select, *processing_controls)
        pagination_controls_top = make_component_pagination_controls()
        pagination_controls_bottom = make_component_pagination_controls()
    pick_buttons = row(export_picks_button, clear)

    update_analysis_control_visibility()
    update_view()
    refresh_status()

    if mode == "single":
        layout = column(controls, p, pick_buttons, status)
    else:
        layout = column(
            controls,
            pagination_controls_top,
            stacked_panel,
            pagination_controls_bottom,
            pick_buttons,
            status,
        )

    if doc is not None:
        doc.add_root(layout)

    return layout


def show(
    pick_state,
    data=None,
    trace_index=0,
    station=None,
    channel=None,
    component=None,
):
    """Display the single-trace picker.

    The widget renders one trace at a time, selected by ``trace_index`` after
    optional station/channel/component filtering. If more than one trace is in 
    the stream, the user can select other traces from a drop down.

    Selectors accept ObsPy wildcard patterns. ``component`` matches the final
    channel letter, while ``channel`` matches the complete channel code.

    Examples
    --------
    show(pick_state, "regional.mseed", station="ANMO", trace_index=0)
    show(pick_state, "regional.mseed", station="ANMO", component="Z")
    show(pick_state, "regional.mseed", station="ANMO", channel="BH*")
    show(pick_state, "station.sac", channel="BHZ")
    """
    def app(doc):
        doc.config.notify_connection_status = False  # no need to see this

        make_arrival_picker_app(
            doc,
            pick_state,
            data=data,
            trace_index=trace_index,
            mode="single",
            station=station,
            channel=channel,
            component=component,
        )

    return _display_bokeh_app(app)

def show_components(
    pick_state,
    data=None,
    station=None,
    channel=None,
    component=None,
):
    """Display filtered traces/components as vertically stacked waveforms.

    Up to 15 traces are displayed at once. Previous/Next controls page through
    larger streams. The filtered data must contain at least two traces. This
    interface does not include a single-trace or display-mode selector.

    Selectors accept ObsPy wildcard patterns. ``component`` matches the final
    channel letter, while ``channel`` matches the complete channel code.

    Examples
    --------
    show_components(pick_state, "regional.mseed", station="ANMO")
    show_components(pick_state, "regional.mseed", station="ANMO", channel="BH*")
    show_components(pick_state, "regional.mseed", station="ANMO", component="Z")
    show_components(pick_state, "path/to/sac_directory", station="ANMO")
    """
    
    def app(doc):
        doc.config.notify_connection_status = False  # no need to see this

        make_arrival_picker_app(
            doc,
            pick_state,
            data=data,
            trace_index=0,
            mode="components",
            station=station,
            channel=channel,
            component=component,
        )

    return _display_bokeh_app(app)