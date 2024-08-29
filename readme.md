Steps to run the application: <br>
<ol>
<li>Flash the <code>motion_and_presence_detection_demo.release.appimage</code> appimage to the device using any tool.
<li>The executable is found in the location <code>$ROOT/exe/lowpower_demo_visualizer_6432_people_counting.exe</code>  
<li>Change the inputs to the application:
    <ul>
    <li>Configuration file: demo files have been provided (<code>cfg_file_application_4m*.cfg</code>). The rangeSelCfg determines the max distance the program uses to count people.
    <li>params file: Contains application and algorithm parameters. The comments in the <code>params.txt</code> file are self-explainable. These need to be changed accordingly.
    <li>save folder (optional): This is the folder where the heatmaps will be saved if <code>saveBool</code> is enabled in the <code>params.txt</code> file.
    </ul>
<li>On clicking the <b>Done</b> button, a figure window will pop-up. 
<br><b>NOTE: It may take some time to display the contents of this window.</b>
</ol>