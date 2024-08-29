clear, clc;
outsideRun = 1;
folderName = "C:/Users/a0491005/Documents/TI/lowpowerpeoplecounting/PeopleCountingExperimentsGaurang/dataHeatmap/data_June2024_Anushree1DCapon";
classList = dir(folderName);
accuracyTable = [];
for i = 1:size(classList,1)
    if strcmp(classList(i).name,".") || strcmp(classList(i).name,"..")
        continue
    end
    testFolder = sprintf("%s/%s",folderName,classList(i).name);
    class = str2num(classList(i).name);
    testFiles = dir(testFolder);
    for j = 1:size(testFiles,1)
        if strcmp(testFiles(j).name,".") || strcmp(testFiles(j).name,"..")
            continue
        end
        testFile = sprintf("%s/%s",testFolder,testFiles(j).name);
        lowpower_demo_visualizer_6432_people_counting;
        accuracyTable = [accuracyTable; class str2num(extractBetween(testFile,strlength(testFile)-7,strlength(testFile)-4)) accuracy];
    end
end


