import cards
import cv2
import argparse
import os
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image file")


# Execute the parse_args() method
args = vars(ap.parse_args())

if not os.path.exists(args["image"]):
    print("ERROR: Path does not exist")
    exit()

# ==============CONSTANTS================
prev = time.time()
IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
font = cv2.FONT_HERSHEY_SIMPLEX

# Template for rank and suit
rankTemplates = cards.load_ranks('Card_Imgs')
suitTemplates = cards.load_suits('Card_Imgs')
print("Loaded templates")

# Read image
image = cv2.imread(args["image"])
IMG_WIDTH = 1280
# IMG_WIDTH = image.shape[1]
IMG_HEIGHT = (image.shape[0] // image.shape[1]) * IMG_WIDTH
image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
# image = cv2.resize(image, IMG_SIZE)

cv2.imshow('Original image', image)

# Preprocess image
thresholded_image = cards.preprocess_image(image)

cv2.imshow('thresholded_image', thresholded_image)

# Find and sort contours of all cards in the image
sorted_contours, contour_is_card = cards.find_cards(thresholded_image)
cardList = []
if len(sorted_contours) != 0:
    # Initialize a new list of card objects
    # k indexes the newly made list of cards
    cardList = []
    k = 0
    # For each contour detected
    for i in range(len(sorted_contours)):
        if(contour_is_card[i] == 1):
            cardList.append(cards.preprocess_card(sorted_contours[i], image))

            # Find the best match for the card
            cardList[k].best_rank_match, cardList[k].best_suit_match, cardList[k].rank_diff, cardList[k].suit_diff = cards.match_card(cardList[k], rankTemplates=rankTemplates, suitTemplates=suitTemplates)
            image = cards.draw_results(image, cardList[k])
            k += 1

temp = image.copy()  
if len(cardList) != 0:
    for i in range(len(cardList)):
        # Draw the card contour
        cv2.drawContours(temp, [cardList[i].contour], 0, (0, 255, 0), 3)
        # Draw the card center
        cv2.circle(temp, (cardList[i].center[0], cardList[i].center[1]), 5, (255, 0, 0), -1)

print(f"Total number of cards found: {len(cardList)}")
print(f"Time used : {time.time() - prev}")
# print(cardList[0])
cv2.putText(temp, f"Number of cards: {len(cardList)}", (10, 50), font, 1, (0, 200, 200), 1, cv2.LINE_AA)
cv2.imshow('result', temp)
cv2.imshow('warped', cardList[0].warp)
cv2.waitKey(0)

