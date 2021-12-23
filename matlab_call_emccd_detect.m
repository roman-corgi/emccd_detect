%pyversion('C:\Users\Kevin\Anaconda3\pythonw.exe');
pyversion;  %For me on my Windows machine, I use the Anaconda distribution of Python,
            %so I open an Anaconda prompt, start MATLAB with the command
            %"matlab", and then all my installed Python libraries and modules, even
            %ones from JPL, work.
%Note that MATLAB 2021b works with Python 3.9 and below, but previous versions use
%Python 3.8 or lower.  And the MATLAB website says it doesn't work with
%Windows Store Python.
%pyenv("ExecutionMode","OutOfProcess");
%If you edit a Python file and save it while this MATLAB file is open, you
%will need to close MATLAB and restart it in order for those changes to
%take effect.

%PYpath = fileparts(which('emccd_detect.py'));
PYpath = fullfile(fileparts(pwd),'emccd_detect');
if count(py.sys.path,PYpath) == 0
    insert(py.sys.path,int32(0),PYpath);
end
%py.importlib.import_module('emccd_detect');
% Input fluxmap of your choosing
fits_name = 'ref_frame.fits';
fits_path = fullfile(pwd, 'data', fits_name);
fluxmaps = (fitsread(fits_path));  % Input fluxmap (photons/pix/s)

%pyOut = py.example_script.emccd_detect(fluxmap=fluxmaps,frametime=100.,em_gain=5000.)
%pyOut = pyrunfile(fullfile(fileparts(pwd),'example_script.py'),'sim_old_style');
%imagesc(sim_old_style)
%fluxmap = py.example_script.fluxmap;

%if you leave out "fits_path=fits_path," below, there's a default file in
%the package that the function will use.

pyOut = py.example_script_m.read_func(fits_path=fits_path,frametime=100, em_gain=5000.,...
        full_well_image=60000.,status=1,dark_current=0.0028, cic=0.02,... 
        read_noise=100.,bias=10000.,qe=0.9,cr_rate=0.,pixel_pitch=13e-6,...  
        eperdn=1.,nbits=64,numel_gain_register=604,choice='legacy');
imagesc(pyOut)
%choice='latest' for latest version of emccd_detect.  For latest,
%typically, eperdn=7, nbits=14.  With eperdn=1 and nbits=64, that's the
%closest you get to getting an output like the legacy one.

%See notes in example_script_m.py for other outputs that are possible. 

% The value for eperdn (electrons per dn) is hardcoded to 1. This is for
%     legacy purposes, as the version 1.0.1 implementation output electrons
%     instead of dn.
% 
%     The legacy version also has no gain register CIC, so cic_gain_register is
%     set to 0 and numel_gain_register is irrelevant.
% 
%     The legacy version also had no ADC (it just output floats), so the number
%     of bits is set as high as possible (64) and the output is converted to
%     floats. This will still be different from the legacy version as there will
%     no longer be negative numbers.

%shot_noise_on is a legacy parameter that has no effect now.  
