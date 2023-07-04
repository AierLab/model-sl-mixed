# Split Learning

Separate a single model into several parts, deploy on different devices but infer/train as a single one.

# Key design

- use `flask` server as the server.
- use `http/https` as protocol.
- serialise or deserialise with `pickle` package.
- encoded byte information in `json` communication.

## Implemented Features

- [x] Communication between split client and split server.
- [x] Forward propagation of intermediate model results.
- [x] Backward propagation of model gradients.
- [ ] Reinforcement learning and corresponding data loaders and model_train function.
- [ ] Lora personalisation option.
- [ ] Abstract everything.
- [ ] Handle exceptions.
