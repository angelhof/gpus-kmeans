# Hipeac GPUs K-means challenge

# Optimization Tasks

Please take over as many tasks as you can

## Low level
- (Sakkas) cuSparse on n x k sparse array
- Point size may not fit in GPU ram
- Research cuda streams 
- Compute squared distance with cublas
- Check if cublas blocks or needs Synchronize 
- Make sure all threads are active at all times (Check block and grid size)
- (Sklikas) Reduce used registers number
- (Sklikas) Change where centers are saved in memory (Textures, constant)

## Report
- (Tsatiris) First Draft

## Evaluation
- (Kallas) Find dataset with many dimension 
- Keep the Final error metric for each implementation, because different ones mean different time to converge
- Make sure that in the final evaluation all implementations have the same tolerance, max\_iters, etc...
- Make sure that the serial k-means is the simple version and doesn't do some kind of optimization

### Automate evaluation
- **(OK)** Write a script that executes all implementations for all datasets
- (Kallas) Improve the script by keeping the size and dimensionality of each dataset so that we can understand the efficiency of the algorithm for different parametrizations
- Integrate the script with the gpu implementations of the algorithm

## Algorithm
- (Greg) Simulated Annealing
- Boruvka
- Yin Yang Kmeans
