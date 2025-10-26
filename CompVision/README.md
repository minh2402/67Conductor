# 67Conductor
Repository for HackNotts 67Conductor

## Documentation
* /lib contains .task & pic
* if you want more than 2 person, or 2 hands. change the options num_poses=2

### setup
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh #Linu
uv init app
uv run main.py
uv add mediapipe
```

```sh
uv sync
uv run main.py
```