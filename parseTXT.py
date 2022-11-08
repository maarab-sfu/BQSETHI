import numpy as np
import scipy.io as spio

def findAccInd(theList):
    for i in range(len(theList)):
        if((theList[i] == "'accuracy':")or(theList[i] == "{'accuracy':")):
            return i+1
    return -1

def main():
    fname = "robustness\lossUP.txt"
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    num = len(content)
    max_data = np.zeros(num)
    acc = np.zeros(num)
    numOfIter = np.zeros(num)
    
    allList = content[0].rsplit(' ')
    print(allList)
    
    for i in range(num):
        myList = content[i].rsplit(' ')
        accInd = findAccInd(myList)
##        print(myList)
##        print(accInd)
##        print(myList.shape)
        max_data[i] = myList[0]
        
        ##Finding accuracy
        myString = myList[accInd]
        x = myString.find('}')
        if(x == -1):
            acc[i] = myString[:-1]
        else:
            acc[i] = myString[:x-1]
        ##Finding the iteration number
        myString = myList[-1]
        x = myString.find('}')
        numOfIter[i] = myString[x+1:]

    x = max_data.argsort()
    max_data = max_data[x]
    acc = acc[x]
    numOfIter = numOfIter[x]

    max_data, x = np.unique(max_data, return_index = True)
    acc = acc[x]
    numOfIter = numOfIter[x]

##    max_data = np.delete(max_data, 17)
##    max_data = np.delete(max_data, 17)
##
##    acc = np.delete(acc, 17)
##    acc = np.delete(acc, 17)
##
##    numOfIter = np.delete(numOfIter, 17)
##    numOfIter = np.delete(numOfIter, 17)
    
    print(max_data, acc, numOfIter)
##    print(np.where(max_data == 599400))
    spio.savemat('KSC_maxData.mat', {"max_data" : max_data})
    spio.savemat('KSC_acc.mat', {"acc" : acc})
    spio.savemat('KSC_numOfIter.mat', {"numOfIter" : numOfIter})


if __name__ == "__main__":
    main()
