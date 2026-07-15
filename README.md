# Interactive Seismogram Widget

An interactive seismogram widget for Jupyter notebooks using [Bokeh](https://github.com/bokeh/bokeh). This widget was originally built as a tool for Earthscope's [Seismology Skill Building Workshop](https://www.earthscope.org/education/skill-building-learning/courses/ssbw/).

For information on how to use it, please see [the tutorial notebook](./interactive-seismogram-widget/tutorial.ipynb).

## Geolab
To use the widget and view the tutorial in [GeoLab](https://www.earthscope.org/data/geolab/), launch GeoLab, sign up for an account or log in using your existing account, and start a GeoLab server from the Hub Control Panel. If you need more help with GeoLab, please see [the documentation](https://docs.earthscope.org/geolab/getting-started).

Once your server is running, you should be at the Launcher. From the Launcher, open a terminal (first choice under "Other"; bottom left icon with a `$_` icon). From the Terminal, download the files by copying and pasting the following command and pressing enter:

```bash
curl -L https://github.com/EarthScope/interactive-seismogram-widget/archive/refs/heads/main.tar.gz \
  | tar -xz --strip-components=1 \
    interactive-seismogram-widget-main/interactive-seismogram-widget
```

While in the same directory as `interactive_seismogram_widget.py`, it can be imported in python: 
```python
import interactive_seismogram_widget
```

## Contributions
To contribute to this tool, please [fork](https://github.com/EarthScope/interactive-seismogram-widget/fork) this repository and open a pull request with your changes.