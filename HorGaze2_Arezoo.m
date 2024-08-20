%
%    HorGaze.m        
%    Horizontal gaze shifts to a single target with varying initial eye and head positions
%
%    call: [H,G,W] = HorGaze([Ton,RetH], E0, H0, DelE, DelH);
%
%     Ton = target onset in ms, RetH = horizontal retinal target location
%     (between [-70, + 70] deg )
%
%      [H,G] = head and gaze trajectories    W: extracted parameters
%    
%                 Arezoo Alizadeh 25-10-2021
%

%     SC burst depends on the initial eye position in the direction of gaze
%
function [spikes_neurons, GAZE, Gvel, HEAD,HF2_Hvel,Evel,EYE,HEAD_contri] = HorGaze2(DELHead,gH0,outSpi_array,TAR, E0, DELEye)

dt=0.00001
t = 0:dt:0.6-dt;     % time in s
Nt = length(t);      % default is 1.0 second time axis a single gaze shift

    %%%%%%%%%%%%% parameters gaze-control model %%%%%%%%%%%%
    
a=20; b=1.5;  % amplitude-duration regression coeffs. for SC burst:   D = a + bR in ms
EBGain = 60;  % linear eye-burst generator gain  
HBGain = 20;  % linear head-burst generator gain

EPAUSE = 40;  % omnipause bias eye in deg/s
HPAUSE = 20;  % omnipause bias head in deg/s

OR=30;            % oculomotor range in deg

% gH0 = 0.5;         % should also depend on E0
gEye = 0.0015;    % small influence of eye-burst output on the head trajectory.
% SCBurst = zeros(1,Nt);  
% SCOUT = zeros(1,Nt);

W = zeros(1,10);

%%%%%%%%%%%%   start the simulation and stimulation %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      if nargin<8
% %         E0_default = [-30:10:30]
% %         DelHead_default = [4000,3000,2000,1000,500,250,150]
% %         p = polyfit(E0_default,DelHead_default,10);
% %         DELHead = polyval(p,E0);  % ms  should also depend on E0 (eye position at gaze onset; will be made noisy)
%         DELHead = 100
%      end
     if nargin<7
        DELEye = 2000;   %   default value; 
%         DELEye = 0;   % ms  default value; 

    end
%     if nargin<7
        H0=0;
%     end
    if nargin<6
        E0=0;       
    end
    if nargin<5
        TAR = [7560, 15];     % retinal target re. the fovea (= E0+T deg re head), presented at 750 ms
    end
           
    EYE = zeros(1,Nt);      % eye position as function of time 
    EYE(1)   = E0;          % the initial eye position
    HEAD = zeros(1,Nt);
    HEAD(1)  = H0;          % the initial head position
    
    Ton = TAR(1);           % stimulus onset time
    
%
%  initialize eye- head velocities and positions
%
Evel=zeros(1,Nt); Hvel=zeros(1,Nt); 
GErr=zeros(1,Nt); HErr=zeros(1,Nt); 
VG=zeros(1,Nt); % vestibular gain
DesEye = zeros(1,Nt); 
eye_me = zeros(1,Nt);
eye_brst=zeros(1,Nt); 
DesHead = zeros(1,Nt); 

EYEON = Ton + DELEye; % a (uniform)random delay [0, 50 ms] for the offset of the pause neurons (= eye saccade onset)
HEADON = Ton + DELHead; % head onset also has some (Gaussian) random value

gH = gH0*(1.0 + tanh(0.05*E0));    % initial eye-position dependent gain for the head displacemnt

W(1) = Ton; W(2) = E0; W(3) = H0; W(4) = gH; W(5) = SCON; W(6) = EYEON; W(7) = HEADON;
%
% start the time loop:  detect stimulus and generate gaze ....
%

%% SC burst
N=200;
pos = linspace(0,5,N);
Bu = 1.4;
Bv = 1.8;
k = 4.3*(10^-6);
A = 3;

rate = zeros(1,length(t));
rate_neurons = zeros(200,length(t));
spikes = zeros(1,length(t));
spikes_neurons = zeros(200,length(t));

for i = 1:200
    M = k*(A*(exp(pos(i)/Bu)-1));
    st = outSpi_array{i,2};

%     unNum   = unique(round(st*10/dt));
%     [n,bin] = histc(round(st/dt),unNum);

%     rate(unNum) = n/dt;

    rate(round(st*1/dt))=1./dt;
    rate_neurons(i,:) = rate*M;
    rate = zeros(1,length(t));

    spikes(round(st*1/dt))=1;
    spikes_neurons(i,:) = spikes;
    spikes = zeros(1,length(t));

end

S_t = sum(rate_neurons)/N;

 for n = 2:Nt
    S_disp(n) = sum(S_t(1:n),2);
 end


for n = 2:Nt     
   
%       if n == TAR(1)  % stimulus onset detected! 
%          STIMON = n;  % this can also be achieved through a differentiator operation as in the paper
%       end
   
%       if n == SCON         % at SC onset          
%          EON = EYE(n-1);   % eye position at sc burst onset 
%               
       %   near saccade onset: calculate the target's desired gaze coordinates.
%           AMP = TAR(2);           % the gaze error is the retinal error
%           HAMP = gH *(AMP + EON);   % desired head amplitude
%          GDur = b * AMP + a + 0.05 * E0; % burst duration (initial eye-position dependent)
%        SCPeak = AMP / (GDur * 0.001);    % in deg/s, also depends on eye position  
%          S= find(S_sgf ~= 0);
%          GDur = S(end) - S(1)
      
%       end
           
       % build the (square-wave) SC burst:    
%        if n >= SCON && n < SCON + GDur
%                       SCBurst(n) = SCPeak;   
%                       SCOUT(n) = SCOUT(n-1) + S_t(n);  % integrated burst output in deg
%                       GErr(n) = SCOUT(n);
%         end  
       %
       %  the gaze error difference-integrator; 
       %
     if n >= EYEON || n >= HEADON   % start the movement(s)     
                  
       GErr(n) = GErr(n-1) + S_t(n-1) - (Evel(n-1) + Hvel(n-1)) * dt;  % gaze error
       VG(n) = VORgain(GErr(n)); 
       %
       %   OMR limitation: calculate desired eye position in the head
       %
       DesEye(n) = omr(EYE(n-1),GErr(n),OR);  
       %
       %   eye motor error and burst - velocity
       %
       eye_me(n) = DesEye(n) - EYE(n-1);
       
       if n>=EYEON
          eye_brst(n) = EBGain * eye_me(n);     % linear burst generator for the eye: eye moves
                      
       else
          eye_brst(:,n)=0;           % eye doesn't move (yet, or anymore)
         
       end
       
       %
       %    cutoff burst eye velocity by OPNs
       %   
       if abs(eye_brst(n))<=EPAUSE 
           eye_brst(n)=0; 
       end
       %
       % The real eye-in-head velocity: with VOR subtracted
       %
         Evel(n) = eye_brst(n) - VG(n)* Hvel(n-1);   % this goes to the gaze feedback     
       %
       %    the actual eye position
       %
       EYE(n) = Evel(n)* dt + EYE(n-1);      % Eye integrator this goes to eye position feedback

    else
        GErr(n) = GErr(n-1) + S_t(n-1) ; 
        %
        %    vor gain and eye position
        %
        VG(n) = 1;
        EYE(n) = EYE(n-1);
    end
    %
    %   the head-movement controller
    %   current head error: current gaze-error+eyepos:
    %
    HErr(n) = GErr(n) +  EYE(n-1); 
    DesHead(n) = gH * HErr(n) - gEye*Evel(n);         % Desired head displacement from G&vO1997 
%     DesHead(n) = gH * HErr(n) ;         % Desired head displacement from G&vO1997 
   
    if DesHead(n)<0
        DesHead(n) = 0;

    end
    
    if n>=HEADON
        Hvel(n) = HBGain * DesHead(n);  % apply LP filter for smoothing; this head-velocity signal goes to the VOR and gaze feedback ...
        
    else
        Hvel(n) = 0;
    end
    if abs(Hvel(n))<=HPAUSE 
        Hvel(n)=0; 
    end
     

HEAD(n) = HEAD(n-1) + Hvel(n)*dt;    % this goes to the target updating circuit
GAZE(n) = EYE(n)+HEAD(n);
     
end   % end time loop

GAZE_off = find(GAZE==max(GAZE(100:end)));
HEAD_contri = HEAD(GAZE_off(1));

start = 1;   stop = length(t);

figure(1); clf;
plot(t(start:stop)-t(start),S_disp(start:stop),'k-','linewidth',1);  hold on
plot(t(start:stop)-t(start),SCOUT(start:stop),'k--','linewidth',1);  hold on
plot(t(start:stop)-t(start),GAZE(start:stop),'k-','linewidth',3);
plot(t(start:stop)-t(start),HEAD(start:stop),'r-','linewidth',2);
plot(t(start:stop)-t(start),EYE(start:stop),'g-','linewidth',2);
plot(t(start:stop)-t(start),10*VG(start:stop),'c-','linewidth',2);
plot(t(start:stop)-t(start),GErr(start:stop),'m-','linewidth',2);
legend('SC','SCOUT','GAZE_S','HEAD_S','EYE_H','10VG', 'GERR');

axis([0 t(stop)-t(start) 0 60]);
xlabel('TIME [s]','fontsize',18);
ylabel('HEAD_S/EYE_H/GAZE_S POSITION [deg]','fontsize',18);
set(gca, 'fontsize',16);

% low pass filter
t2 = [0:1:12000];
T1 = 2500; T2 = 1e-6; 
h = (1/(T1-T2))*(exp(-t2/T1)-exp(-t2/T2));
HF2_Hvel = conv(Hvel,h); 
HF2_Hvel = HF2_Hvel(1:length(Hvel));

% T1 = 2500; T2 = 1e-6; 
% h = (1/(T1-T2))*(exp(-t2/T1)-exp(-t2/T2));
% HF2_Evel = conv(Evel,h); 
% HF2_Evel=HF2_Evel(1:length(Evel));
Evel = sgolayfilt(Evel,1,501)
% Gvel = Hvel + Evel;
Gvel = HF2_Hvel + Evel;

figure(2); clf
plot(t(start:stop)-t(start), abs(HF2_Hvel(start:stop)),'r-','linewidth',2);
hold on
plot(t(start:stop)-t(start)-0.005, abs(Gvel(start:stop)),'k-','linewidth',2);
plot(t(start:stop)-t(start),Evel(start:stop),'g-','linewidth',2);


axis([0 t(stop)-t(start) -100 800]);
legend('HEAD_S','GAZE_S','EYE_H');
xlabel('TIME [s]','fontsize',18);
ylabel('HEAD_S/EYE_H/GAZE_S VELOCITY [deg/s]','fontsize',18);
set(gca, 'fontsize',16);


return
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   
%%%%%%%%%%% function definitions 1D gaze control model %%%%%%%%%%%%%%%%%%%%%%%

function ed = omr(eye,tar,OR)        % oculomotor range limitations

    beta = 0.03;
    % calculate the desired end position of the eye
    TH = tar + eye;
    ed = OR*tanh(beta*TH);
   

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vg = VORgain(GE)
   alfa = 0.03;
   GEamp = abs(GE);
   % gain varies between 0 (high GEamp) and 1 (low GEamp)
   vg = 1.0 - tanh(alfa*GEamp);
   return
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
   