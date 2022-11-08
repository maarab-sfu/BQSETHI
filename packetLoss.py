def main():
    # [ 8  3 15 62 14]
    # (834136, 211384, 54760, 14464, 3616)
    qualities = [8,  3, 15, 62, 14]
    oneHotQuality = np.concatenate((np.ones(qualities[0]), np.ones(qualities[1]), np.ones(qualities[2]), np.ones(qualities[3]), np.ones(qualities[4])))
    sizes = [834136, 211384, 54760, 14464, 3616]
    max_data = np.sum(np.multiply(qualities, sizes))
    packetSize = 1000
    numOfPackets = max_data // packetSize
    for i in range(12):
        numOfLoss = i*numOfPackets // 100
        theZeros = np.zeros(numOfLoss)
        theOnes = np.ones(numOfPackets - numOfLoss)
        indices = np.concatenate((theZeros, theOnes))
        for j in range(30):
            indices = np.random.shuffle(indices)
            print(indices)
            for k in range(indices):
                if(indices[k] == 0):
                    if(k <= qualities[0]*sizes[0])
                        for k1 in range(qualities[0]):
                            if(k<=k1*sizes[0]):
                                oneHotQuality[k1] = 0
                
                    
                    
            
    packetLossindex = np.random.randint(2, size = numOfPackets)
