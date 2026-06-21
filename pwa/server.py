"""Simple HTTP server for EdgeHealth PWA. Run: python server.py"""
import http.server
import socket

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.', **kwargs)

# Get local IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print(f"EdgeHealth PWA Server")
print(f"  Local:  http://localhost:{PORT}")
print(f"  iPhone: http://{local_ip}:{PORT}")
print(f"  (Both devices must be on same WiFi!)")

http.server.HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
