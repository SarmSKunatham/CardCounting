import numpy as np
import cv2
import time
import os

# ============= CONSTANT ================

# Adaptive threhold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width & height of card corner, where rank and suite are
CORNER_WIDTH = 28
CORNER_HEIGHT = 84

# Dimensions of rank in the template images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit in the template images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 20000

font = cv2.FONT_HERSHEY_SIMPLEX

# ============= STRUCTURE ================
class QCard:

    '''Structure to store information about cards from the image'''

    def __init__(self):
        self.contour = []
        self.width = 0
        self.height = 0
        self.corner_points = [] # Corner point
        self.center = [] # Center point
        self.warp = [] # Perspective transform
        self.rank_img = []
        self.suit_img = []
        self.best_rank_match = "Unknown"
        self.best_suit_match = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0
        self.bounding_rect = [] # Bounding rectangle

    def __str__(self) -> str:
        return f"Card:\nWidth: {self.width}\nHeight: {self.height}\nCorner points: {self.corner_points}\nCenter: {self.center}\nWarp: {self.warp}\nRank image: {self.rank_img}\nSuit image: {self.suit_img}\nBest rank match: {self.best_rank_match}\nBest suit match: {self.best_suit_match}\nRank difference: {self.rank_diff}\nSuit difference: {self.suit_diff}\nBounding rectangle: {self.bounding_rect}"

class RankTemplate:

    '''Structure to store information about rank template images'''

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from harddrive
        self.name = ""

class SuitTemplate:

    '''Structure to store information about suit template images'''

    def __init__(self):
        self.img = [] # Thresholded, sized suit image loaded from harddrive
        self.name = ""

# ============= FUNCTIONS ================

def load_ranks(filepath):
    '''
    Input:
        Filepath of the rank image directory
    Return:
        List of RankTemplate Obj.
    '''

    Ranks = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
    rankTemplates = []

    for rank in Ranks:
        rankCardTemplate = RankTemplate()
        rankCardTemplate.img = cv2.imread(os.path.join(filepath, rank + '.jpg'), 0)
        rankCardTemplate.name = rank
        rankTemplates.append(rankCardTemplate)

    return rankTemplates

def load_suits(filepath):
    '''
    Input:
        Filepath of the suit image directory
    Return:
        List of SuitTemplate Obj.
    '''

    Suits = ['Spades', 'Diamonds', 'Clubs', 'Hearts']
    suitTemplates = []

    for suit in Suits:
        suitCardTemplate = SuitTemplate()
        suitCardTemplate.img = cv2.imread(os.path.join(filepath, suit + '.jpg' ), 0)
        suitCardTemplate.name = suit
        suitTemplates.append(suitCardTemplate)

    return suitTemplates

def preprocess_image(image):
    '''
    Input:
        An BGR image
    Return:
        A Grayed blurred and adaptive thresholded image
    '''
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur image with gaussian blue to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using Otsu's method to find the optimal threshold value then apply it to get the binarized image
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def find_cards(thresh_image):
    '''
    Input:
        - An thresholded image
    Return:
        - Number of cards found 
        - List of sorted contours of cards from largest to smallest
    '''

    # Find contours and sort them by size
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If there is no contour, do nothing
    if len(contours) == 0:
        return 0, []

    # Sort there indices by contour size
    index_sort = sorted(range(len(contours)), key=lambda k: cv2.contourArea(contours[k]), reverse=True)

    # Initialize empty sorted contour and hierarchy lists
    sorted_contours = []
    sorted_hierarchy = []
    is_card = np.zeros(len(contours), dtype=np.uint8)

    # Fill the sorted contour and hierarchy lists
    # The hierarchy list can be used to check if the contours have parents or not
    for i in index_sort:
        sorted_contours.append(contours[i])
        sorted_hierarchy.append(hierarchy[0][i])

    # Check which contour is a card by the criteria
    # 1. Smaller area than CARD_MAX_AREA
    # 2. Larger area than CARD_MIN_AREA
    # 3. Has no parent contour
    # 4. Has 4 corners
    for i in range(len(contours)):
        size = cv2.contourArea(sorted_contours[i])
        perimeter = cv2.arcLength(sorted_contours[i], True)
        approx = cv2.approxPolyDP(sorted_contours[i], 0.01 * perimeter, True)

        # print(sorted_hierarchy[i])

        if (( size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and (sorted_hierarchy[i][3] == -1) and (len(approx) == 4)):
            print(f"Size: {size}")
            print(f"Card found at index {i}")
            print("Approx: ", approx)
            is_card[i] = 1

    return sorted_contours, is_card


def preprocess_card(contour, image):
    '''
    Use contour to find information about the card. Get the rank and suit image from the card.
    Input:
        - A contour
        - An image
    Return:
        - A card object
    '''
    # Initialize a card object
    Card = QCard()

    # Contour
    Card.contour = contour

    # Find the perimeter of card and approximate the corner points
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    points = np.float32(approx)
    Card.corner_points = points

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    Card.bounding_rect = [x, y, w, h]
    Card.width = w
    Card.height = h
    
    # Center point, taking x and y average of the four corners.
    average = np.mean(points, axis=0)
    Card.center = [int(average[0][0]), int(average[0][1])]

    # Warp the card by perspective transform
    Card.warp = four_point_transform(image, points)

    # Grab corner of warped card and zoom in
    cardCorner = Card.warp[0: CORNER_HEIGHT, 0: CORNER_WIDTH]
    cardCornerZoom = cv2.resize(cardCorner, (0, 0), fx=4, fy=4)
    cv2.imshow('cardCornerZoom', cardCornerZoom)

    # # Sample known white pixel intensity to determine the threshold level
    # # white_level = cardCornerZoom[15, int((CORNER_WIDTH * 4) / 2)]
    # # thresh_level = white_level - CARD_THRESH

    # # Threshold the card corner
    # # if (thresh_level <= 0):
    # #     thresh_level = 1
    _, cornerThresh = cv2.threshold(cardCornerZoom, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('ThreshCorner', cornerThresh)

    # # Split into top and bottom halves : top half is the rank, bottom half is the suit
    rankThresh = cornerThresh[20: 185, 0: 128]
    suitThresh = cornerThresh[186: 336, 0: 128]
    cv2.imshow('rankThresh', rankThresh)
    cv2.imshow('suitThresh', suitThresh)

    # Find rank contour and bounding rectangle, isolate and find largest contour
    rankContours, hierarchy = cv2.findContours(rankThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rankContours = sorted(rankContours, key=cv2.contourArea, reverse=True)
    # Find bounding rectangle for largest contour
    if len(rankContours) > 0:
        x1, y1, w1, h1 = cv2.boundingRect(rankContours[0])
        rankROI = rankThresh[y1: y1 + h1, x1: x1 + w1]
        rankResized = cv2.resize(rankROI, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        Card.rank_img = rankResized
        cv2.imshow('rankResized', rankResized)

    # Find suit contour and bounding rectangle, isolate and find largest contour
    suitContours, hierarchy = cv2.findContours(suitThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    suitContours = sorted(suitContours, key=cv2.contourArea, reverse=True)
    # Find bounding rectangle for largest contour
    if len(suitContours) > 0:
        x2, y2, w2, h2 = cv2.boundingRect(suitContours[0])
        suitROI = suitThresh[y2: y2 + h2, x2: x2 + w2]
        suitResized = cv2.resize(suitROI, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        Card.suit_img = suitResized
        cv2.imshow('suitResized', suitResized)

    return Card

def order_points(points):
    '''
    Input:
        - A list of 4 points
    Return:
        - A list of 4 points, sorted from top-left to bottom-right
    '''
    print(f"Points: {points}")
    rect = np.zeros((4, 2), dtype=np.float32)
    # s = points.sum(axis = 1)
    s = np.sum(points, axis=2)
    print(f"s: {s}")
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis = -1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect

def four_point_transform(image, points):
    '''
    Input:
        - An image
        - Four corner points
        - Width and height of the card
    Return:
        - A warped image
    '''
    # Sort the points to make sure the order is top-left, top-right, bottom-right, bottom-left
    rect = order_points(points)
    # print(np.sum(points, axis=2))
    print(f'rect: {rect}')
    maxWidth = 200
    maxHeight = 300
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ],
        dtype = "float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    return warped

def match_card(Card, rankTemplates, suitTemplates):
    '''
    Finds the best match for rank and suit. The best match is the template with the smallest difference.
    Input:
        - A card object
        - A list of rank templates
        - A list of suit templates
    Return:
        - A card object with rank and suit
    '''
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    # If no contours are found, return unknown
    if Card.rank_img is None or Card.suit_img is None:
        Card.rank = "Unknown"
        Card.suit = "Unknown"
        return Card
    if (len(Card.rank_img) != 0 and len(Card.suit_img) != 0):
        # Difference between card and rank template, store the result with the smallest difference
        for rankTemplate in rankTemplates:
            cv2.imshow(f"rankTemplate {i}", rankTemplate.img)
            diff_img = cv2.absdiff(Card.rank_img, rankTemplate.img)
            rank_diff = int(np.sum(diff_img) / 255.0)

            if rank_diff < best_rank_match_diff:
                # best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = rankTemplate.name
        
        # Same for the suit
        for suitTemplate in suitTemplates:
            diff_img = cv2.absdiff(Card.suit_img, suitTemplate.img)
            suit_diff = int(np.sum(diff_img) / 255.0)

            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_name = suitTemplate.name
        
        if (best_rank_match_diff < RANK_DIFF_MAX):
            best_rank_match_name = best_rank_name
        if (best_suit_match_diff < SUIT_DIFF_MAX):
            best_suit_match_name = best_suit_name

    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff
    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)
    
    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #s_diff = str(qCard.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

