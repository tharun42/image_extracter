import cv2
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

original_image = cv2.imread("group.jpg")




# Define boundary rectangle containing the foreground object
def drawLine(event,x,y,flags,params):
    global ixLineAdd,iyLineAdd,ixLineRemove, iyLineRemove, x_coord_img, y_coord_img, draw_flag, color, whichClick, imageHeight, imageWidth, lineAdd, lineRemove
    # Left Mouse Button Down Pressed
    if(event==1):
        draw_flag = True
        color = 255
        whichClick = 1
        x_coord_img = x
        y_coord_img = y
    # Right mouse button Down Pressed
    if(event==2):
        draw_flag = True
        color = 0
        whichClick = 2
        x_coord_img = x
        y_coord_img = y

    if(event==0):
        if(draw_flag==True):
            #For Drawing Line
            cv2.line(imgTmp,pt1=(x_coord_img,y_coord_img),pt2=(x,y),color=(color,color,color),thickness=3)
            x_coord_img = x
            y_coord_img = y
            if(whichClick == 1):
                for i in range(0, 5):
                    if(((x_coord_img+i) < imageWidth) and ((x_coord_img-i) >= 0)):
                        #print(x_coord_img+i,i)
                        ixLineAdd.append(x_coord_img+i)
                    if(((y_coord_img+i) < imageHeight) and ((y_coord_img-i) >= 0)):    
                        iyLineAdd.append(y_coord_img+i)
                    if(((x_coord_img-i) >= 0) and ((x_coord_img+i) < imageWidth)):
                        ixLineAdd.append(x_coord_img-i)
                    if(((y_coord_img-i) >= 0) and ((y_coord_img+i) < imageHeight)):
                        iyLineAdd.append(y_coord_img-i)
            elif(whichClick==2):
                for i in range(0, 5):
                    if(((x_coord_img+i) < imageWidth) and ((x_coord_img-i) >= 0)):
                        #print(x_coord_img+i,i)
                        ixLineRemove.append(x_coord_img+i)
                    if(((y_coord_img+i) < imageHeight) and ( (y_coord_img-i) >= 0)):
                        iyLineRemove.append(y_coord_img+i)
                    if(((x_coord_img-i) >= 0) and ((x_coord_img+i) < imageWidth)):
                        ixLineRemove.append(x_coord_img-i)
                    if(((y_coord_img-i) >= 0) and ((y_coord_img+i) < imageHeight)):
                        iyLineRemove.append(y_coord_img-i)
                    #print(ixLineRemove,i)
                
    # Mouse button released
    if(event==4 or event==5):
        draw_flag = False
        if(len(iyLineAdd)>0 and len(ixLineAdd)>0):
            n = min(len(iyLineAdd), len(ixLineAdd))
            lineAdd = np.vstack((iyLineAdd[:n], ixLineAdd[:n])).T
        if(len(iyLineRemove)>0 and len(ixLineRemove)>0):
            n = min(len(iyLineRemove), len(ixLineRemove))
            lineRemove = np.vstack((iyLineRemove[:n], ixLineRemove[:n])).T
        
def drawRect(event,x,y,flags,params):
    global x_coord_img,y_coord_img,ix2,iy2,draw_flag
    # Left Mouse Button Down Pressed
    if(event==1):
        draw_flag = True
        x_coord_img = x
        y_coord_img = y
    # Mouse button released
    if(event==4):
        draw_flag = False
        cv2.rectangle(imgTmp,pt1=(x_coord_img,y_coord_img),pt2=(x,y),color=(255,255,255),thickness=3)
        ix2=x
        iy2=y

draw_flag = False
x_coord_img, y_coord_img, ix2, iy2 = -1, -1, -1, -1
imgTmp = original_image.copy()
originalImage = original_image.copy()
# Making Window For The Image
cv2.namedWindow("Select the foreground with a rectangle",cv2.WINDOW_NORMAL)

# Adding Mouse CallBack Event
cv2.setMouseCallback("Select the foreground with a rectangle",drawRect)

# Starting The Loop So Image Can Be Shown and allow exit on "Q" key pressed
while(True):
    cv2.imshow("Select the foreground with a rectangle",imgTmp)
    key = cv2.waitKey(20) & 0xFF
    if key == 13:  
        break
cv2.destroyAllWindows()
figure, ax = plt.subplots(1)

# Rectangle info for foreground, conditions so the rectangle works in every way
if(ix2 <x_coord_img):
    ix2, x_coord_img = x_coord_img, ix2
if(iy2<y_coord_img):
    iy2, y_coord_img = y_coord_img, iy2
# Save the rectangle's information
width = abs(ix2-x_coord_img)
height = abs(iy2-y_coord_img)
xStart = x_coord_img
yStart = y_coord_img

# Show rectangle including foreground item on image
rect = patches.Rectangle((x_coord_img ,y_coord_img),width ,height , edgecolor='r', facecolor="none")
ax.imshow(original_image)
ax.add_patch(rect)
rect = (xStart,yStart,width,height)

# Set the seed for reproducibility purposes
cv2.setRNGSeed(0)

# Initialize GrabCut mask image, that will store the segmentation results

number_of_iterations = 15

mask = np.zeros(original_image.shape[:2], np.uint8)
mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_PR_FGD

background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

cv2.grabCut(
    original_image,
    mask,
    rect,
    background_model,
    foreground_model,
    number_of_iterations,
    mode=cv2.GC_INIT_WITH_RECT,
)

grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(
    "uint8"
)
new_mask = np.where((grabcut_mask == cv2.GC_PR_BGD) | (grabcut_mask == cv2.GC_BGD), 0, 1).astype(np.uint8)

assert original_image.shape[:2] == new_mask.shape[:2]
assert original_image.dtype == new_mask.dtype

binary_mask = np.where((grabcut_mask == cv2.GC_PR_BGD) | (grabcut_mask == cv2.GC_BGD), 0, 1).astype(np.uint8)

binary_mask = cv2.merge((binary_mask, binary_mask, binary_mask))


segmented_image = original_image * binary_mask
cv2.imwrite("segmented_image.png", segmented_image)

cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
