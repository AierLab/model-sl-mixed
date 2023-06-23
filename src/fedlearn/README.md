# Federated Learning

## Environment Setup
1.Install [flower](https://flower.dev/docs/install-flower.html#install-stable-release)
  ```
  python -m pip install flwr 
  ```
2.Install Pytorch 
  ```
  python -m pip install torch torchvision
  ```
## Run Federated Learning
  ```
  cd path/to/this/folder
  ```
  Start the server first.
  ```
  python server.py
  ```
  Then open a new terminal to start the first client.
  ```
  cd path/to/this/folder
  python client.py
  ```
  Repeat the previous step in the third terminal to start the second client. Normal simulation requires at least two clients, you can add as many clients as you want.
  
  
### You can configure the server using following command line arguments：

| argument | description |
| --------------- | --------------- |
| **-r** \<rounds\> | Number of rounds for the federated training. Defaults to 3.|
| **-fc** \<fit clients\> | Min fit clients, min number of clients to be sampled next round. Defaults to 2.|
| **-ac** \<available clients\> | Min available clients, min number of clients that need to connect to the server before training round can start. Defaults to 2.|
|**-ckpt** \<checkpoint\> | Path to checkpoint to be loaded. Defaults to "".|
|**-addr** \<address\> | Server address. Defaults to "0.0.0.0:8080".|

## Implemented Features

- [x] Federated learning + Pytroch.
- [x] Load checkpoint on server side.
- [x] Save checkpoint on client side.

## TODOs
- [ ] Save checkpoint on server side.