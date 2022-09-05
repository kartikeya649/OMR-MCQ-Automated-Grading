# OMR-MCQ-Automated-Grading
Final Year Project

This is my final year AI based project.
Optical mark recognization (OMR) MCQ Automated Grading is a AI model which is Python & Open CV based which can run on any python compiler.  By using this model we can scan omr sheet and grade them. It takes as input an image of an answered answer sheet process the sheet and give outputs which alternatives were marked and also grade the omr sheet 

By Using this model user can easily scan the omr sheet from dataset and gets the grade of that omr sheet on basis of the comparison with reference omr sheet which user set as reference omr sheet.
In this project i used open cv from ai domain 
There are very simple steps omr scanning.
Step 1: First we take an original image or image from dataset.
Step 2:  convert the taken image in grey scale.
Step 3:  Find the edges in grey scaled image using cunny.
Step 4:  Find the contour present in grey scaled image.
Step 5:  find the biggest rectangle present in the image and their corner points
Step 6:  Then we take wrap IPU which is wrap perspective and later apply some threshold on the image.
Step7: We find the marks where each correct mark are present.
Step 8: lastly, we save the outputs.

In the future, we can provide user authentication to the system so that we have data about who has done the scanning of the images and if something has happened in an illegal manner we can reach out to the concerned user. With this, we can allow access to the system only to the users who are trained in this software.
