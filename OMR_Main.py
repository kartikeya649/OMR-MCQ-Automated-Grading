import cv2
import numpy as np
import utlis

path = "9.jpg"


heightImg = 700
widthImg  = 700
questions=5
choices=5
ans= [1,2,2,2,4]


img = cv2.imread(path)

#preprocessing
img=cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContour=img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 


#finding All Contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

#Find Rectangles
rectCon = utlis.rectCountour(contours) # FILTER FOR RECTANGLE CONTOURS
biggestContour= utlis.getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
gradePoints = utlis.getCornerPoints(rectCon[1]) # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContour, biggestContour, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    cv2.drawContours(imgBiggestContour, gradePoints, -1, (255, 0, 0), 20) # DRAW THE BIGGEST CONTOUR

    biggestContour=utlis.reorder(biggestContour) # REORDER FOR WARPING
    gradePoints = utlis.reorder(gradePoints) # REORDER FOR WARPING

    pt1 = np.float32(biggestContour) # PREPARE POINTS FOR WARP
    pt2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pt1, pt2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE

    ptG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)# GET TRANSFORMATION MATRIX
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # APPLY WARP PERSPECTIVE


    #apply threshold
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE


    boxes=utlis.splitBoxes(imgThresh)
    #cv2.imshow("Test",boxes[2])
    #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

    myPixelVal = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
    countR=0
    countC=0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC]= totalPixels
        countC += 1
        if (countC==choices):countC=0;countR +=1
    

    # FIND THE USER ANSWERS AND PUT THEM IN A LIST
    myIndex=[]
    for x in range (0,questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    #print(myIndex) 

     # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
    grading=[]
    for x in range(0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:grading.append(0)
    #print("GRADING",grading)
    score = (sum(grading)/questions)*100 # FINAL GRADE
    print("SCORE",score)   

    # DISPLAYING ANSWERS
    imgResult=imgWarpColored.copy()
    utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices) # DRAW DETECTED ANSWERS
    #imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)  # DRAW DETECTED ANSWERS
    imRawDrawings = np.zeros_like(imgWarpColored) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
    utlis.showAnswers(imRawDrawings,myIndex,grading,ans,questions,choices)
    #imRawDrawings = utlis.showAnswers(imRawDrawings, myIndex, grading, ans, questions, choices)

    invMatrix = cv2.getPerspectiveTransform(pt2, pt1) # INVERSE TRANSFORMATION MATRIX
    imgInvWarp = cv2.warpPerspective(imRawDrawings, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP


    imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # NEW BLANK IMAGE WITH GRADE AREA SIZE
    cv2.putText(imgRawGrade,str(int(score))+"%",(70,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3) # ADD THE GRADE TO NEW IMAGE
    #cv2.imshow("Grade",imgRawGrade)
    invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1) # INVERSE TRANSFORMATION MATRIX
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # INV IMAGE WARP

    # SHOW ANSWERS AND GRADE ON FINAL IMAGE
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)
    


imgBlank = np.zeros_like(img)
imageArray=([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContour,imgWarpColored,imgThresh],
            [imgResult,imRawDrawings,imgInvWarp,imgFinal])

 # LABELS FOR DISPLAY
lables = [["Original","Gray","Blur","Canny"],["Contours","Biggest Con","Warp","Threshold"],["Result","Raw Drawaing","inv Wrap","Final"]]
imgStacked=utlis.stackImages(imageArray,0.3,lables)


cv2.imshow("Final Result",imgFinal)
cv2.imshow("Stacked Images",imgStacked)
cv2.waitKey(0)