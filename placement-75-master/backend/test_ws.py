import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/gd_module/ws/gd_meeting/1"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
           
            # Receive the first message
            resp1 = await websocket.recv()
            print("Msg 1:", resp1)

            # Receive the next message (maybe the bot starts speaking?)
            resp2 = await websocket.recv()
            print("Msg 2:", resp2)
           
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())