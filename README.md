# QWERTY - CENG 596 Term Project 


## Tweet Categories
Each tweet category is represented by a code, which is a number between 0 and 7, and label such as Music or Health. 
Available tweet categories are the followings:
* 0 TV&Movies
* 1 Music
* 2 Health
* 3 Religion
* 4 Politics
* 5 Technology
* 6 Sports
* 7 Other

## Tweets
**tweets.txt** contains to tweets used to train and test our classifier. The following style is used to append new tweets to this file:
```bash
<Tweet_HERE>
$$$$$<Tweet_Code_Here>
```
Note that tweets should be in English.

## Preprocessing
Preprocessing steps which are applied to each tweet during the classifications are the followings:
* Stemming
* Removing usernames
* Removing hashtags
* Removing hyperlinks
* Removing numeric characters
* Removing punctuation
* Removing single characters
* Removing emojis
* Removing English stop words

## Web Application
This project has a simple web application where you can check the category of the tweet.
Once you run app.py, you can access web application on http://localhost:5000/