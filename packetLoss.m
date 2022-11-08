clear
clc
qualities = [0,  0, 75, 0, 0];
oneHotQuality = [ones(1,qualities(5)), ones(1,qualities(1)), ones(1,qualities(2)), ones(1,qualities(3)), ones(1,qualities(4))];
sizes = [834136, 211384, 54760, 14464, 3616];
max_data = sum(qualities.*sizes)
packetSize = 1000;
numOfPackets = fix(max_data/packetSize);
bandSizes = fix(sizes/packetSize);

for i = 1:12
    filename = "UP_loss"+int2str(i)+".txt"
    fileID = fopen('robustness\loss\'+filename, 'a');
    numOfLoss = fix(i*numOfPackets/100);
    theZeros = zeros(1, numOfLoss);
    theOnes = ones(1, (numOfPackets - numOfLoss));
    indices = [theZeros theOnes];
    for j = 1:10
            indices = indices(randperm(length(indices)));
            ind = 0;
            while(ind<length(indices))
                ind = ind + 1;
                if(indices(ind) == 0)
                    [quality, number] = findzero(ind, qualities, bandSizes);
                    fprintf(fileID, '%d %d\n', quality, number);
                end
            end
    end
    fclose(fileID)
end

function [quality, number] = findzero(ind, qualities, bandSizes)
    offset = 0;
    for qual = 1:5
        for i = 1:qualities(qual)
            if(ind <= i*bandSizes(qual) + offset)
                quality = qual;
                number = i;
                return
            end
        end
        offset = offset + qualities(qual)*bandSizes(qual);
    end
    quality = 5;
    number = qualities(5);            
end
