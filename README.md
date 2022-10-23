The zip file contains y2raytracingproj as the main testing script.

All other python files are modules that are imported onto the test script in the first code cell.

In the test script, I have hashed all the plt.savefig() lines. If at any point you do want to unhash these, please note that plt.show() lines would require hashing.

If you are running on older python which relies returns all plots in the console, 
for the xy plots, feel free to change the figure sizes of the plots if they are too large for the console.
You can find the two methods for the xy plotting in raytracer.py and ray_bundle.py

List of files submitted:
y2raytracingproj.py
raytracer.py
OpticalElement1.py
SphericalRefraction.py
output_plane.py
ray_bundle.py
README.txt 
