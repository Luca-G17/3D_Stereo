# 3D From Stereo
[Full Report](../main/report.pdf)
### Hough Circles:
This was a Univeristy project aimed at depth identification via stereo imagery. In this case the images are generated with open3D and features consist of uniform spheres distributed throughout the scene. The first step was to identify
the 2D locations of each sphere in each camera's view plane.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/54ed54e3-1d81-4dc0-90a9-397016751a25" alt="alt text" width="320" height="240">
  <img src="https://github.com/user-attachments/assets/53105e7d-aeb2-4650-b239-1e341210ca67" alt="alt text" width="320" height="240">
<p align="center">
  
### Correspondence Matching:  
We can then cast epipolar lines into the scene, from the focal point of each camera through the centre of each sphere on the viewplane. Pairs of epipolar lines from opposing cameras which have the shortest perpendicular distance are assumed to
intersect at the 3D centre of their corresponding sphere.

$$\lbrace(L_i, L_j) | \min{(\text{perpdist}(L_i, L_j))}\rbrace = \text{Corresponding epipolar lines}$$   
<p align="center">
  <img src="https://github.com/user-attachments/assets/74dcdd97-9ea7-473b-ad14-3a774aa01826" alt="alt text" width="320" height="240">
  <img src="https://github.com/user-attachments/assets/4844dc71-4e09-4c04-9405-3720d35254c9" alt="alt text" width="320" height="240">
<p align="center">

Finally we can use the computed depth to the centre of each sphere and it's corresponding circles' radius' on the 2D view plane to estimate the radius of each sphere in 3D space. 

$$\frac{Zr}{f}=R=\text{Sphere Radius} \qquad Z=\text{Depth} \qquad f=\text{Focal Length} \qquad r=\text{Circle Radius}$$
<p align="center">
  <img src="https://github.com/user-attachments/assets/4226d6d3-a7a2-4255-b91b-32f13b6d3656" alt="alt text" width="641" height="240">
<p align="center">
