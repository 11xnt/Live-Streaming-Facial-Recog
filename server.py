# Imports
import sys
from livestreaming import app
import socketio
from waitress import *
import logging

# Logging is enabled to help
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# create a socket io server
sio = socketio.Server()
# create a wsgi application using the socket io and application
appServer = socketio.WSGIApp(sio, app)


if __name__ == '__main__':
    try:
        logger.info("Server starting")
        serve(appServer, host='', port=5000, url_scheme='http', threads=6, expose_tracebacks=True,
              log_untrusted_proxy_headers=True)
    except KeyboardInterrupt:
        logger.info("Server Shutting down")
        sys.exit(0)
