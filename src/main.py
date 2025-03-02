import asyncio
from camera_trap import CamTrap
import os

#
# get file and path details
#

current_directory = os.getcwd()
path = os.path.abspath(os.path.join(current_directory, 'outputs/images'))

#
# Setup Camera Trap System
#

camtrap = CamTrap(path, duration=300)

#
# Start session
#

asyncio.run(camtrap.run())

print('Ending recording sessions...')
