import asyncio
from camera_trap import CamTrap
import datetime
import os

#
# get file and path details
#

current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%S")
current_directory = os.getcwd()
path = os.path.abspath(os.path.join(current_directory, 'outputs/images'))
file = f"{current_date}"
path = path + '/' + file
duration = 10

#
# Setup Camera Trap System
#

camtrap = CamTrap(path, duration)

#
# Start session
#

asyncio.run(camtrap.run())

print('Ending recording sessions...')
