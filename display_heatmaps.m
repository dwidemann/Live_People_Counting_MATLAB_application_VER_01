for i = 5:239
    imagesc(fHist(i).rangeAzimuthHeatMapMinor);
    title(sprintf('%d',i))
    pause(.1);
end
close();