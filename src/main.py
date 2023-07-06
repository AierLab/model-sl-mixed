from expose import Server
from model import ChatModel

if __name__ == '__main__':
    chat_model = ChatModel("")
    server = Server(chat_model.process)
    server.run("localhost", 10086)

