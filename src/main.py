from expose import Server
from model import ChatModelDemo

if __name__ == '__main__':
    chat_model = ChatModelDemo()
    server = Server(chat_model.process)
    server.run("localhost", 10086)

