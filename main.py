import cv2
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

# Read the original image
IMAGES_FOLDER = "D:\Camera Roll"
original_image = cv2.imread("11.jpg")

#detecting edges.
#gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
#edges = cv2.Canny(gray, threshold1=30, threshold2=100)
figure, ax = plt.subplots(1)
ax.imshow(original_image)


# Define boundary rectangle containing the foreground object
def drawLine(event,x,y,flags,params):
    global ixLineAdd,iyLineAdd,ixLineRemove, iyLineRemove, ix, iy, drawing, color, whichClick, imageHeight, imageWidth, lineAdd, lineRemove
    # Left Mouse Button Down Pressed
    if(event==1):
        drawing = True
        color = 255
        whichClick = 1
        ix = x
        iy = y
    # Right mouse button Down Pressed
    if(event==2):
        drawing = True
        color = 0
        whichClick = 2
        ix = x
        iy = y

    if(event==0):
        if(drawing==True):
            #For Drawing Line
            cv2.line(imgTmp,pt1=(ix,iy),pt2=(x,y),color=(color,color,color),thickness=3)
            ix = x
            iy = y
            if(whichClick == 1):
                for i in range(0, 5):
                    if(((ix+i) < imageWidth) and ((ix-i) >= 0)):
                        #print(ix+i,i)
                        ixLineAdd.append(ix+i)
                    if(((iy+i) < imageHeight) and ((iy-i) >= 0)):    
                        iyLineAdd.append(iy+i)
                    if(((ix-i) >= 0) and ((ix+i) < imageWidth)):
                        ixLineAdd.append(ix-i)
                    if(((iy-i) >= 0) and ((iy+i) < imageHeight)):
                        iyLineAdd.append(iy-i)
            elif(whichClick==2):
                for i in range(0, 5):
                    if(((ix+i) < imageWidth) and ((ix-i) >= 0)):
                        #print(ix+i,i)
                        ixLineRemove.append(ix+i)
                    if(((iy+i) < imageHeight) and ( (iy-i) >= 0)):
                        iyLineRemove.append(iy+i)
                    if(((ix-i) >= 0) and ((ix+i) < imageWidth)):
                        ixLineRemove.append(ix-i)
                    if(((iy-i) >= 0) and ((iy+i) < imageHeight)):
                        iyLineRemove.append(iy-i)
                    #print(ixLineRemove,i)
                
    # Mouse button released
    if(event==4 or event==5):
        drawing = False
        if(len(iyLineAdd)>0 and len(ixLineAdd)>0):
            n = min(len(iyLineAdd), len(ixLineAdd))
            lineAdd = np.vstack((iyLineAdd[:n], ixLineAdd[:n])).T
            #iyLineAdd = np.unique(iyLineAdd, axis=0)
            #ixLineAdd = np.unique(ixLineAdd, axis=0)
        if(len(iyLineRemove)>0 and len(ixLineRemove)>0):
            n = min(len(iyLineRemove), len(ixLineRemove))
            lineRemove = np.vstack((iyLineRemove[:n], ixLineRemove[:n])).T
            #iyLineRemove = np.unique(iyLineRemove, axis=0)
            #ixLineRemove = np.unique(ixLineRemove, axis=0)
        
def drawRect(event,x,y,flags,params):
    global ix,iy,ix2,iy2,drawing
    # Left Mouse Button Down Pressed
    if(event==1):
        drawing = True
        ix = x
        iy = y
    # Mouse button released
    if(event==4):
        drawing = False
        cv2.rectangle(imgTmp,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
        ix2=x
        iy2=y

drawing = False
ix, iy, ix2, iy2 = -1, -1, -1, -1
imgTmp = original_image.copy()
originalImage = original_image.copy()
# Making Window For The Image
cv2.namedWindow("Select the foreground with a rectangle")

# Adding Mouse CallBack Event
cv2.setMouseCallback("Select the foreground with a rectangle",drawRect)

# Starting The Loop So Image Can Be Shown and allow exit on "Q" key pressed
while(True):
    cv2.imshow("Select the foreground with a rectangle",imgTmp)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
figure, ax = plt.subplots(1)

# Rectangle info for foreground, conditions so the rectangle works in every way
if(ix2 <ix):
    ix2, ix = ix, ix2
if(iy2<iy):
    iy2, iy = iy, iy2
# Save the rectangle's information
width = abs(ix2-ix)
height = abs(iy2-iy)
xStart = ix
yStart = iy

# Show rectangle including foreground item on image
rect = patches.Rectangle((ix ,iy),width ,height , edgecolor='r', facecolor="none")
ax.imshow(original_image)
ax.add_patch(rect)
rect = (xStart,yStart,width,height)

# Set the seed for reproducibility purposes
cv2.setRNGSeed(0)

# Initialize GrabCut mask image, that will store the segmentation results

number_of_iterations = 5
#gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
#show_image(gray_image, "Gray image")



# Initialize the mask with known information
mask = np.zeros(original_image.shape[:2], np.uint8)
mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_PR_FGD

#mask[binarized_image == 0] = cv2.GC_FGD

#edge_mask = np.zeros_like(mask)
#edge_mask[edges > 0] = cv2.GC_FGD

#mask |= edge_mask

# Arrays used by the algorithm internally
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

# Ensure that original_image and new_mask have the same dimensions and data type
assert original_image.shape[:2] == new_mask.shape[:2]
assert original_image.dtype == new_mask.dtype

# Verify the data type of new_mask (should be np.uint8)
print("Data type of new_mask:", new_mask.dtype)

# Create a binary mask (0 or 1) based on grabcut_mask values
binary_mask = np.where((grabcut_mask == cv2.GC_PR_BGD) | (grabcut_mask == cv2.GC_BGD), 0, 1).astype(np.uint8)

# Verify the shape and data type of binary_mask
print("Shape of binary_mask:", binary_mask.shape)
print("Data type of binary_mask:", binary_mask.dtype)

# Expand the binary mask to 3 channels
binary_mask = cv2.merge((binary_mask, binary_mask, binary_mask))

# Verify the shape and data type of the binary_mask after expansion
print("Shape of binary_mask (after expansion):", binary_mask.shape)
print("Data type of binary_mask (after expansion):", binary_mask.dtype)


segmented_image = original_image * binary_mask
cv2.imwrite("segmented_image.png", segmented_image)

# Display the segmented image
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()