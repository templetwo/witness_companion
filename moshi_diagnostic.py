#!/usr/bin/env python3
"""
moshi_diagnostic.py - WebSocket Message Inspector
==================================================
Connects to Moshi and shows EXACTLY what format it sends.
This helps us build the proper parser.

Usage:
1. Start Moshi: python -m moshi_mlx.local_web
2. Run this: python moshi_diagnostic.py
3. Talk to Moshi and watch the console
"""

import asyncio
import json
import websockets


async def inspect_moshi():
    uri = "ws://localhost:8998/api/chat"

    print("\n" + "="*60)
    print("   MOSHI WEBSOCKET DIAGNOSTIC")
    print("="*60 + "\n")
    print(f"Connecting to: {uri}")
    print("Speak to Moshi and watch the message format...\n")
    print("-"*60)

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to Moshi\n")

            message_count = 0
            text_tokens = []

            while True:
                try:
                    message = await websocket.recv()
                    message_count += 1

                    # Moshi sends binary messages with type prefix byte
                    if isinstance(message, bytes) and len(message) > 0:
                        msg_type = message[0]
                        payload = message[1:]

                        if msg_type == 0x00:
                            print(f"[#{message_count}] HANDSHAKE (0x00)")

                        elif msg_type == 0x01:
                            # Audio - just count, don't spam
                            if message_count % 50 == 0:
                                print(f"[#{message_count}] AUDIO (0x01) - {len(payload)} bytes")

                        elif msg_type == 0x02:
                            # Text token
                            try:
                                text = payload.decode('utf-8')
                                text_tokens.append(text)
                                print(f"[#{message_count}] TEXT (0x02): '{text}'")

                                # Show accumulated sentence
                                combined = "".join(text_tokens)
                                if any(combined.rstrip().endswith(p) for p in '.!?'):
                                    print(f"\n  ➜ Complete: \"{combined.strip()}\"")
                                    text_tokens = []
                                    print("-"*60)
                            except:
                                print(f"[#{message_count}] TEXT (0x02): [decode error]")

                        else:
                            print(f"[#{message_count}] UNKNOWN (0x{msg_type:02x})")

                    else:
                        print(f"[#{message_count}] Non-binary message: {type(message)}")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"\n⚠ Error receiving: {e}")
                    break

    except ConnectionRefusedError:
        print("\n✗ Cannot connect to Moshi!")
        print("Make sure Moshi is running:")
        print("  python -m moshi_mlx.local_web --no-browser")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(inspect_moshi())
    except KeyboardInterrupt:
        print("\n\n[Diagnostic stopped]")
