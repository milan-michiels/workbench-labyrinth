# Vids

The first, second and third use as the observation space the balls coordinates and the velocity of the ball.

## First

The first batch of videos marked by "-first" are videos with some random actions given by gymnasium. This was to test if
it was possible to move and record.

## Second

The second batch of videos marked by "-second" are videos of a SAC agent not trained on the environment.

## Third

The third batch of videos marked by "-third" are videos of a SAC agent trained on the environment. The videos are not
the training but the evaluation of the agent.

## Fourth

The fourth batch of videos marked by "-fourth" are videos of a SAC agent trained on the environment. The videos are not
the training but the evaluation of the agent. The difference with the third batch is that the agent gets a different
observation space. Instead of the ball coordinates and the velocity of the ball, the agent gets a 8x8 array of the 
environment.

## Latest

The latest batch of videos marked by "-latest" are videos of a SAC agent trained on the environment. The videos are not
the training but the evaluation of the agent. The difference with the fourth batch is that the agent gets a higher resolution
of the environment. Instead of the 8x8 array, the agent gets a 8x8 grayscale image of the environment.