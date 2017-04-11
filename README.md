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
- (Kallas) Automate evaluation 

## Algorithm
- Simulated Annealing
- Boruvka
- Yin Yang Kmeans
