
## Example assessment code

Here is an example evaluation code, which is similar to how we evaluate your agent in Gradescope.
This can be also found at the end of [my_agent.py](../my_agent.py).
```python
# the below resembles how we evaluate your agent
env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

episodes = 10
scores = list()
for episode in range(episodes):
    env2.play(player=agent2)
    scores.append(env2.score)

print(np.max(scores))
print(np.mean(scores))
```

**The score here is defined as the number of pipes disposed after they moved out of the screen from the left hand side.**
As you can see, your agent will have 10 chances at each difficulty level (per submission). 
We will evaluate the agent using `mode=eval` and load your model weight file.
Please name the weight file `my_model.ckpt` and submit it together with your `my_agent.py` and `pytorch_mlp.py` 
(even if you do not change the content). No other files should be uploaded, nor they will be used even if you choose to update them.
Finally, similar to assignment 1, please define the following variables in your 
[my_agent.py](../my_agent.py).

```python
STUDENT_ID = 'a1234567'
DEGREE = 'UG' # or 'PG'
```

## Assessment criteria
The assessment criteria are defined below:
* Lv 1 (8 marks) - your agent achieves an average score of 10 (here game_length = 10).
* Lv 2 (7 marks) - your agent achieves an average score >= 5 (here game_length = 10).
* Lv 3 (6 marks) - your agent's highest score is >= 5 (here game_length = 10).
* Lv 4 (UG) / Lv 5 (PG) (6 marks) - your agent's highest score is >= 5 (here game_length = 10).
* Lv 6 (3 marks) - this one is a challenger level, your mark is calculated as:
`mark = min(ceil(((highest_score - 1) / 10) * 3), 3)`. This means you need to achieve a minimum score of 2 to get one mark, and achieves a minimum of 8 to get the full three marks. During the evaluation we will set game_length = 50 for this level. 

**After each level is completed the marking script should run the next level without rerunning the passed level(s).
Note we have to find out if Gradescope can support such testing scheme to finalise this part, stay tuned!**

## Submission 
You must submit your program files on the Gradescope portal linked in MyUni.
The code will be compiled and run on the above tests. If it passes all tests, you will receive 30% of the overall course mark. There are no manual marks for style or commenting.
However, we will inspect your submission should Gradescope flag your submission with high code similarity to other submissions.

## Using other source code
You may not use other source codes for this assignment. You should personally and carefully implement the algorithm to fully understand the concept.

## Due date
This assessment's due date is specified on MyUni.

## Portals:
1. [Main page](../README.md)
2. [Installation and playing the game](INSTALLATION.md)
3. [Assignment description](ASSIGNMENT_DESCRIPTION.md)
4. [Game information](GAME_INFORMATION.MD)
5. Assessment description <- you are here
6. [Helpful advice](HELPFUL_ADVICE.md)