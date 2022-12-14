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
        # empty 'host' will run default on ip address, set value to localhost if needed
        # threads value equates to given
        serve(appServer, host='0.0.0.0', port=8080, url_scheme='http', threads=6, expose_tracebacks=True,
              log_untrusted_proxy_headers=True)
    except KeyboardInterrupt:
        logger.info("Server Shutting down")
        sys.exit(0)
