"# Detecting-Hashtag-Hijacking-for-Hashtag-Activism" 

This code is a semi-supervised framework to detect Tweet-level hash-tag hijacking targeting specific social movements,using a combination of features based on the Tweettext, user profile and timeline, replies, and hash-tag coherence. We focus on #MeToo movement,but our methodology can be applied to any othermovement or hashtag. Most prior work on hashtaghijacking has focused on general trending hashtagslike #job or #android and could not adapt over timeto attacker strategies. These approaches are limitedto specific contexts and do not take into account thechanging characteristics of hashtag use over time.To best of our knowledge, this is the first time thata semi-supervised method is used to detect hashtaghijacking at Tweet level.

Our framework contains two modules including semi-supervised detector and batch model update.

CollectData package contains three python classes: 
1)CollectHistoricalData collects data using tweeter premium account. take into consideration that all keys are removed for security issues. This class collects all metoo tweets from Oct 2017 to Nov 2019 using the procesure  defined in valid Data set of paper.
2) collectsAdditionalData: collects all usertimeline data and hijackeddata following Hijacked data set procodure explained in paper.
3)collectLiveTweets: Using tweeter streaming APi to collect Metoo Tweets each day.

Processing Package contains classes related to cleaning data for processing for further use.

All Module of our semi-supervided module is locate in its corresponding package including:
UserBlackListModule,
WhiteListModule,
Supervised
unsupervised

 Batchupdate module function are also located into the related like updateing hijcaked words dictionary adn updating white user are  located in WhiteListModule and updaing usreblackLis is located in UserBlackListModule. Each class has its own main function for testing and running separately but the main routine of the program is defined in main.py class.



