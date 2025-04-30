## Installation 

Please run the following command in your terminal to install the needed Python libraries. Please note the Numpy version needs to be <2.
If you already have installed the dependent Python libraries prior to this assignment, and you experience errors out of the box running the game, you 
should consider making a new virtualenv or a new Conda environment for this assignment as otherwise it could be a result of mismatching library versions. 
If you do not know how to do that, you may ask ChatGPT "How can I make a virtualenv/conda environment for a new Python project?". 

```bash
pip install -r requirement.txt
```


## Play the game

After installation, you can verify the game is functional by playing it yourself:

```bash
python3 play_game.py --level 1
```

The five levels of the game are:
* Lv1 - "Sky is the limit." 
  * There are "no pipes". As long as the bird does not drop out of the screen, the game continues. However, when you read the emulator fed 
  state information, you will see it does contain pipes after the game starts for a period of time.
* Lv2 - "Easy peasy lemon squeezy!" 
  * The pipe openings (vertical gap between top/bottom pipes) stay level in the middle of the screen, the bird has a narrower space to navigate but it is still straightforward.
* Lv3 - "Life has its ups and downs." 
  * The pipe openings start to go up and down following a sine wave function through time. The agent needs to learn
  to jump strategically to accommodate the pipe opening changes. 
* Lv4 (UG) / Lv5 (PG) - "Life is full of random pipes."
  * The pipe openings are randomly positioned following a uniform distribution. This starts to resemble the original game.
  * Lv4 is for UG and Lv5 is for PG, PG's pipes are slightly wider to mildly increase the difficulty.
* Lv6 - "Birdie thinks the pipes are getting mean!"
  * The pipes appear faster and the opening's vertical gap gets smaller.

The levels are defined in the [config.yml](../config.yml) together with other game parameters.

 ## Portals:
1. [Main page](../README.md)
2. Installation and playing the game <- you are here
3. [Assignment description](ASSIGNMENT_DESCRIPTION.md)
4. [Game information](GAME_INFORMATION.MD)
5. [Assessment description](ASSESSMENT_DESCRIPTION.md)
6. [Helpful advice](HELPFUL_ADVICE.md)