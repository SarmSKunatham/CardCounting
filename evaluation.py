import cards
import cv2
import os
import pandas as pd

filePaths = []
for root, dirs, files in os.walk('cardsdataset/test'):
    for filename in files:
        if filename.endswith('.jpeg'):
            filePaths.append(os.path.join(root, filename))

# ==============CONSTANTS================
IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
font = cv2.FONT_HERSHEY_SIMPLEX

# Template for rank and suit
rankTemplates = cards.load_ranks('Card_Imgs')
suitTemplates = cards.load_suits('Card_Imgs')
print("Loaded templates")
results = []
for filePath in filePaths:

    # Read image
    image = cv2.imread(filePath)
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
        results.append(cardList)

# Store the results in a file
answer = []

for index, result in enumerate(results):
    output = []
    recog = []
    print(len(result))
    output.append(filePaths[index])
    output.append(len(result))
    for index, card in enumerate(result):
        print(card.best_rank_match, card.best_suit_match)
        string = f"{card.best_rank_match} {card.best_suit_match}"
        recog.append(string)
    output.append(recog)
    answer.append(output)

df = pd.DataFrame(answer, columns=['filename','num_cards', 'recog'])
print(df)
df.to_csv('output.csv', index=False)

