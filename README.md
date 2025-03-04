# QCP-Group-Work
This is a repository for our Quantum Computing Project 


add Fourier transform to circuit
have a fast and slow implementation of Grovers to demonstrate speed up
comparing old and new
![image](https://github.com/user-attachments/assets/13c08a38-d295-4e2b-a1a6-b35beee86649)
comparing new for only one iteration for each, bare in mind the iterations scale linearly (for 1 oracle value, 24 is like 500 iterations, each like 10 seconds long)
![image](https://github.com/user-attachments/assets/88c1d691-37eb-42b4-98f3-b57270483b9d)
The limitation at this point is the prob dist, but i have also sped that up a lot now, ironically you need to classical search through every item in the list to find the highest probs, maybe we could implement grovers for the prob list XD

