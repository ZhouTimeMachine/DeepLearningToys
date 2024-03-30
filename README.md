# DeepLearningToys
Toy examples for some deep learning models, selected by [ZhouTimeMachine](https://github.com/ZhouTimeMachine).

Feel free to *issue* if you have any suggestions, or you can contact me in other ways.

It is recommended to use [online documentation](https://zhoutimemachine.github.io/DeepLearningToys/). If you want to deploy and render it locally, you can follow the guidance of [Local Deployment](#local-deployment).

## Content


## Local Deployment

Firstly install [mkdocs](https://www.mkdocs.org/) support.

```
pip install mkdocs
pip install mkdocs-material
pip install mkdocs-heti-plugin
```

Start the real-time rendering service (with default port `8000`)
```
mkdocs serve
```

If everything goes well, enter `127.0.0.1:8000` in the browser to preview locally. However, if port `8000` is occupied, you may need to specify a new port, taking port `8001` as an example:
```
mkdocs serve -a 127.0.0.1:8001
```

At this time, you need to use `127.0.0.1:8001` for local preview.