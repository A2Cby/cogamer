import asyncio
from time import monotonic

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, WebSocketException
from logger import logger
import ssl

class WebSocketClient:
    """
    A WebSocket client that automatically reconnects on connection loss.
    """
    RECONNECT_INTERVAL = 60*1  # seconds

    class WebSocketError(Exception):
        ...


    class WebSocketConnectionClosed(WebSocketError):
        ...

    class WebSocketConnectionError(WebSocketError):
        ...

    def __init__(self, logger, uri: str):
        self._logger = logger
        self._uri = uri
        self._connection: ClientConnection | None = None
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        self._manager_task: asyncio.Task | None = None
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self._last_connection_time: float | None = None

    @property
    def is_connected(self) -> bool:
        """Check if the websocket is currently connected."""
        return self._is_connected



    async def init_connect(self):
        await self.connect()
        print('in connect')
        if self._manager_task:
            self._logger.warning("Connection manager is already running.")
            return
        self._manager_task = asyncio.create_task(self._connection_manager())

    async def connect(self):
        async with self._connection_lock:
            _ssl_context = ssl.create_default_context()
            _ssl_context.check_hostname = False
            _ssl_context.verify_mode = ssl.CERT_NONE
            self._last_connection_time = monotonic()
            self._connection = await websockets.connect(self._uri, ssl=_ssl_context,
                                                        additional_headers={"Content-Type": "application/json"})
            self._is_connected = True
            self._last_connection_time = monotonic()


    async def reconnect(self):
        if not self._manager_task:
            self._logger.warning("Connection manager is not running.")
            return

        self._manager_task.cancel()
        try:
            await self._manager_task
        except asyncio.CancelledError:
            pass
        self._manager_task = None

        async with self._connection_lock:
            if self._connection:
                self._logger.warning("Thy to reconnect.")
                res = await self._connection.close(code=1000, reason="Client shutting down")
                logger.warning(res)
                self._connection = None
                self._is_connected = False
                self._logger.info("WebSocket connection has been disconnected.")
                await self.init_connect()
    async def disconnect(self):
        """Stops the connection manager and closes the connection."""
        if not self._manager_task:
            self._logger.warning("Connection manager is not running.")
            return

        self._manager_task.cancel()
        try:
            await self._manager_task
        except asyncio.CancelledError:
            pass
        self._manager_task = None

        async with self._connection_lock:
            if self._connection:
                await self._connection.close(code=1000, reason="Client shutting down")
                self._connection = None
                self._is_connected = False
                self._logger.info("WebSocket connection has been disconnected.")

    async def send(self, message: str):
        if not self.is_connected or not self._connection:
            self._is_connected = False
            raise self.WebSocketConnectionError("WebSocket is not connected.")

        async with self._connection_lock:
            try:
                res = await self._connection.send(message)
            except ConnectionClosed as e:
                self._is_connected = False
                self._logger.error(f"Failed to send message: Connection was closed. {e}")
                raise self.WebSocketConnectionClosed("Failed to send message: Connection was closed.")

    async def force_send(self, message: str):
            try:
                await self.send(message)
            except (self.WebSocketConnectionError, self.WebSocketConnectionClosed):
                await self.connect()
                await self.send(message)
            except Exception as e:
                self._logger.warning(f"Failed to send message: {message}: {e}")

    async def receive(self) -> str:
        if not self.is_connected or not self._connection:
            raise self.WebSocketConnectionError("WebSocket is not connected.")

        try:
            message = await self._connection.recv()
            # self._logger.info(f"Received message: {message}")
            return message
        except ConnectionClosed:
            self._is_connected = False
            self._logger.error("Failed to receive message: Connection was closed.")
            raise self.WebSocketConnectionClosed("Failed to receive message: Connection was closed.")

    async def force_receive(self) -> str | None:
        try:
            message = await self.receive()
        except (self.WebSocketConnectionError, self.WebSocketConnectionClosed):
            await self.connect()
            message = await self.receive()
        except Exception as e:
            self._logger.warning(f"Failed to receive message: {e}")
            return None
        return message

    async def _connection_manager(self):
        """
        The core background task that maintains and reconnects the connection.
        """
        reconnect_delay = 1  # Start with a 1-second delay

        while True:
            try:
                self._logger.info(f"Attempting to connect to {self._uri}...")
                if not self._is_connected:
                    await self.connect()
                self._logger.info(f"Successfully connected to {self._uri}.")
                reconnect_delay = 1

                await self._connection.wait_closed()

            except (WebSocketException, ConnectionRefusedError, OSError) as e:
                self._logger.error(f"Connection failed: {e}. Retrying in {reconnect_delay} seconds...")

            finally:
                self._is_connected = False
                self._connection = None

                await asyncio.sleep(reconnect_delay)

                # Exponential backoff: increase delay for the next attempt
                reconnect_delay = min(reconnect_delay * 2, 60)  # Cap at 60 seconds



async def main():
    WS_URI = "ws://localhost:8765"

    client = WebSocketClient(uri=WS_URI, logger=logger)
    await client.init_connect()

    await asyncio.sleep(2)

    async def message_sender(client: WebSocketClient):
        counter = 0
        while True:
            if client.is_connected:
                try:
                    await client.send(f"Hello, message number {counter}!")
                    counter += 1
                except ConnectionClosed:
                    client._logger.warning("Sender: Connection lost. Waiting for reconnect.")
            else:
                client._logger.info("Sender: WebSocket is not connected. Waiting...")
            await asyncio.sleep(3)

    async def message_receiver(client: WebSocketClient):
        while True:
            if client.is_connected:
                try:
                    message = await asyncio.wait_for(client.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue  # No message received, continue loop
                except ConnectionClosed:
                    client._logger.warning("Receiver: Connection lost. Waiting for reconnect.")
            else:
                await asyncio.sleep(1)

    # Run sender and receiver tasks concurrently
    sender_task = asyncio.create_task(message_sender(client))
    receiver_task = asyncio.create_task(message_receiver(client))

    # Let it run for a while. Try stopping and restarting your WebSocket server
    # during this time to see the automatic reconnect in action.
    try:
        await asyncio.sleep(120)
    finally:
        sender_task.cancel()
        receiver_task.cancel()
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user.")