# Football match person classification Telegram Bot

This Telegram Bot is solution for <a href="https://challenge.uma.tech/">UmaTech.Challenge</a> hackathon's Machine Learning track.<br>
Bot alias: @FootballUmaTechBot

### Task
Create Telegram Bot that receive photo of person on football match and predict type of that person.

There are 25 types of persons:
 - 11 different type of first team (10 players, 1 goal keeper)
 - 11 different type of second team (10 players, 1 goal keeper)
 - Referee (полевой судья)
 - Linesman (боковой судья)
 - Other
 
 ### Data
 Photos of persons grabbed from 1 match. Each type has 150 photos, so types are balanced.
 
 Photo shapes are around 60 px at height and 30 px at width.
 
**Examples**:
<p float="left">
<img src="https://gitlab.com/MelnikovAlmaz/football_classification_bot/uploads/2761794beeebe31ee9509bad6bb767ad/yellow_goalkeeper.png" height="120px"/>
<img src="https://gitlab.com/MelnikovAlmaz/football_classification_bot/uploads/463c7f6c0a373cd0a9a5cbadac2e3f66/blue_player.png" height="120px"/>
<img src="https://gitlab.com/MelnikovAlmaz/football_classification_bot/uploads/6df42babb65082637755bdb3c78f1848/white_player.png" height="120px"/>
<img src="https://gitlab.com/MelnikovAlmaz/football_classification_bot/uploads/24c16bf03a29b0483e8e651896e7657b/referee.png" height="120px"/>
<img src="https://gitlab.com/MelnikovAlmaz/football_classification_bot/uploads/385777fff318a39e30620ad347fa20da/linesman.png" height="120px"/>
<img src="https://gitlab.com/MelnikovAlmaz/football_classification_bot/uploads/4607e7ec829d703cc7758fa5bb5fb932/other.png" height="120px"/>
</p>


### Classifier
Classifier has hierarchical structure and consists of 4 simple classifiers. <br>
The structure of classifier
------------
    ├── BasicNet - CNN for 6 classes
        ├── White players (10 types merged)
            └── PlayerNet (white players) - CNN for 10 classes
                ├── First white player
                ├── ...
                └── Tenth white player
        ├── Blue players (10 types merged) - CNN for 10 classes
            └── PlayerNet (blue players)
                ├── First blue player
                ├── ...
                └── Tenth blue player
        ├── Yellow goalkeeper
        ├── Green goalkeeper
        ├── Referees (2 types merged)
            └── RefferiNet - CNN for 2 classes
                ├── Refferi
                └── Linesman
        └── Other 
 
 Each network has 2 Conv layers, 2 MaxPooling, 2 FC layers (the last one is classification layer).<br>
 Network structures can be founded in **classifier/networks** folder.<br>
 
 Each network trained on 90% of photos of each class with random shuffle and tested on 10%.
 
 #### Results of training
 **BasicNet**<br>
 Accuracy on training - 96,6%<br>
 Accuracy on test - 96,3%
 
 **ReferriNet**<br>
 Accuracy on training - 100%<br>
 Accuracy on test - 100%
 
 **PlayerNet white team**<br>
 Accuracy on training - 92,8%<br>
 Accuracy on test - 73,8%
 
  **PlayerNet blue team**<br>
 Accuracy on training - 96,8%<br>
 Accuracy on test - 81,1%
 
 ### Appendix
Trainig pipelines code and data are not in this repository.<br>
If you want to reproduce training process please contact me:
 - Email: melnikovalmaz@gmail.com
 
