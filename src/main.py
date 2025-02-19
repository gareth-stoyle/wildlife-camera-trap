from camera_trap import CamTrap
import datetime
import os

#
# get file and path details
#

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_directory = os.getcwd()
path = os.path.abspath(os.path.join(current_directory, 'test_images'))
file = f"{current_date}"
path = path + '/' + file
duration = 10*60

#
# Setup Camera Trap System
#

camtrap = CamTrap(path, duration)

#
# Start session
#

camtrap.run()

print('Ending recording sessions...')
