function [X_mesh,arrival_times,jump_sizes] = ...
    gg_200921_gillespie_gou_2step_8(k,t_matrix,S,nCells,sde_params)
k = repmat(k,[nCells,1]);

kappa = sde_params(1);
lambda = sde_params(2);
beta = sde_params(3);
ALPHA = lambda/kappa;
k(:,1) = gamrnd(ALPHA,1/beta,nCells,1);
num_t_pts = size(t_matrix,2);

tvec = t_matrix(1,:);
T = t_matrix(1,end);

%%%%%%%%%%%%
% EDGES_GAMMA = linspace(0,5,100);
% figure(2)
% 
% % x_ = linspace(0.05,max(k(:,1))); 
% subplot(131);
% H=histogram(k(:,1),300, 'normalization','pdf','FaceColor',0.8*[1,1,1],...
%     'EdgeColor','none'); hold on;
% EDGES_GAMMA = H.BinEdges;
% x_gamma = EDGES_GAMMA(1:end-1) + diff(EDGES_GAMMA)/2;
% x = x_gamma;
% y = gampdf(x,ALPHA,1/beta); %y = y./sum(y);
% plot(x,y,'r--','LineWidth',2);
% set(gca,'yscale','log')

X_mesh = NaN(nCells,num_t_pts,5);  %2 species + 3 parameters

t = zeros(nCells,1); 
tindex = ones(nCells,1);

%initialize state: integer unspliced, integer spliced 
X = zeros(nCells,2);
X_mesh(:,1,1:2) = X;
X_mesh(:,1,3:5) = k;

%initialize list of cells that are being simulated
simindices = 1:nCells;
activecells = true(nCells,1);

%First loop: simulate the underlying Gamma-OU process. 
%The number of arrivals is given by a Poisson RV. 
num_arrivals = poissrnd(T*lambda,nCells,1);
n_arriv_dim = max(num_arrivals);
arrival_times = inf(nCells, n_arriv_dim);
jump_sizes = zeros(nCells, n_arriv_dim);
for i = simindices
    %The arrival times are given by the order statistics of a uniform RV.
    arrival_times(i,1:num_arrivals(i)) = sort(T*rand(1,num_arrivals(i)));
    %The jump sizes are given by an exponential RV.
    jump_sizes(i,1:num_arrivals(i)) = exprnd(1/beta,1,num_arrivals(i));
end

%For computational facility, we append another arrival at zero,
%corresponding to the initial distribution of transcription rates.
arrival_times = [zeros(nCells,1), arrival_times];
num_arrivals = num_arrivals+1;
n_arriv_dim = n_arriv_dim+1;
%To compute the integrals for Gillespie, we need interarrival periods.
interarrival_periods = diff(arrival_times,1,2);
interarrival_periods(~isfinite(interarrival_periods)) = Inf;
jump_sizes = [k(:,1), jump_sizes];

%initialize the arrays to store summary information. These fully determine
%the dynamics of the Gamma-OU process.
magnitude_at_jumps = zeros(nCells, n_arriv_dim);
interval_integrals = zeros(nCells, n_arriv_dim);
magnitude_at_jumps(:,1) = jump_sizes(:,1);
for i = 2:(n_arriv_dim)
    Exp_deg = exp(-kappa*interarrival_periods(:,i-1));
    magnitude_at_jumps(:,i) = magnitude_at_jumps(:,i-1).*Exp_deg ...
        + jump_sizes(:,i);
    %the following line is only valid when Exp_deg is evaluated over a
    %finite inter-arrival period. 
    %therefore, it will always give an erroneous integral for index
    %following the terminal jump. This can be filtered out here, but we
    %take the easier approach of filtering later based on finite arrival
    %time.
    interval_integrals(:,i) = magnitude_at_jumps(:,i-1)/kappa.*(1-Exp_deg);
end
interval_integrals(~isfinite(arrival_times)) = NaN;
jump_sizes(~isfinite(arrival_times)) = NaN;
arrival_times(~isfinite(arrival_times)) = NaN;


for i = 1:nCells
    for j = 1:num_t_pts
%     for j = [1, num_t_pts]
        t__ = tvec(j);
        N_arriv = sum(arrival_times(i,:) <= t__);

        X_mesh(i,j,3) = sum(exp(-kappa*(t__-arrival_times(i,1:N_arriv))) ...
            .* jump_sizes(i,1:N_arriv));
    end
end
 
% figure(3)
% clf;
% II_ = 1;
% for i_ = 1:II_
% %     plot(t_,Kint_val(i_,:),'-','Color',0.7*[1 1 1]); hold on;
%     plot(t_,K_val(i_,:),'-','Color',0.7*[1 1 1]); hold on;
% end
% 
% % plot(t_,mean(K_val,1),'k-','LineWidth',2);
% plot([0,T],[1 1]*ALPHA/beta,'m--');
% 
% % plot(t_,Kint_val(II_,:),'k-'); hold on;
% plot([0,T],[1 1]*mean(K_val(1,:)),'r-','LineWidth',2);
% set(gca,'yscale','log');
% xlabel('time');
% ylabel('transcription rate');
% legend('one realization','mean','theoretical','location','best');
cum_integ = cumsum(interval_integrals,2);
% for i = 1:num_arrivals(II_)
%     plot([0,T],[1,1]*cum_integ(II_,i),'r--');
% end

% 
% figure(2)
% subplot(132);
% H=histogram(x(:,1),300, 'normalization','pdf','FaceColor',0.8*[1,1,1],...
%     'EdgeColor','none'); hold on;
% EDGES_GAMMA = H.BinEdges;
% x_gamma = EDGES_GAMMA(1:end-1) + diff(EDGES_GAMMA)/2;
% y = gampdf(x_gamma,ALPHA,1/beta); %y = y./sum(y);
% plot(x_gamma,y,'r--','LineWidth',2);
% set(gca,'yscale','log')
% 
% subplot(133);
% H=histogram(x(:,end),300, 'normalization','pdf','FaceColor',0.8*[1,1,1],...
%     'EdgeColor','none'); hold on;
% EDGES_GAMMA = H.BinEdges;
% x_gamma = EDGES_GAMMA(1:end-1) + diff(EDGES_GAMMA)/2;
% y = gampdf(x_gamma,ALPHA,1/beta); %y = y./sum(y);
% plot(x_gamma,y,'r--','LineWidth',2);
% set(gca,'yscale','log')
%Second loop: simulate the actual Gillespie
step = 0;
while any(activecells)
    mu = zeros(nCells,1);
    
%     if 
    [t_upd,mu_upd] = rxn_calculator(...
        X(activecells,:),...
        t(activecells),...
        k(activecells,:),...
        sum(activecells),...
        sde_params,...
        magnitude_at_jumps(activecells,:),...
        arrival_times(activecells,:),...
        interval_integrals(activecells,:),...
        num_arrivals(activecells),step);

    
    kold = k;
%     if any(
    k_upd = kinit_eval(sde_params, magnitude_at_jumps(activecells,:),...
        arrival_times(activecells,:),t_upd,sum(activecells));
    k(activecells,1) = k_upd;
    t(activecells) = t_upd;
    mu(activecells) = mu_upd;
    
    
    linindupdate = sub2ind(size(t_matrix),(1:length(tindex(activecells)))',...
        tindex(activecells));
    tvec_time = t_matrix(linindupdate);
    update = false(nCells,1);
    update(activecells) = t(activecells)>tvec_time;
    
    while any(update)
        tobeupdated = find(update);
        for i = 1:length(tobeupdated)
            X_mesh(simindices(tobeupdated(i)),tindex(tobeupdated(i)),1:2) = ...
                X(tobeupdated(i),:);
            X_mesh(simindices(tobeupdated(i)),tindex(tobeupdated(i)),4:5) = ...
                kold(tobeupdated(i),2:3);
            
        end
        tindex = tindex+update;
        ended_in_update = tindex(update)>num_t_pts;

        if any(ended_in_update)
            ended = tobeupdated(ended_in_update);
            
            activecells(ended) = false;
            mu(ended) = 0;

            if ~any(activecells),break;end
        end
        
        linindupdate = sub2ind(size(t_matrix),(1:length(tindex(activecells)))',...
            tindex(activecells));
        tvec_time = t_matrix(linindupdate);
        update = false(nCells,1);
        update(activecells) = t(activecells)>tvec_time;

    end
    
    z_ = find(activecells);
    try
        X(z_,:) = X(z_,:) + S(mu(z_),:);
    catch
        disp('incorrect reaction computation!');
    end
    step = step+1;
end

return


function [t,mu] = rxn_calculator(X,t,k,nCells,sde_params,...
    magnitude_at_jumps,arrival_times,interval_integrals,num_arrivals,step)
% nSteps=100;
% nRxn = 4;
kappa = sde_params(1);
nRxn = 3; 

% a = zeros(nCells,nRxn);

% a is propensity matrix
% reactions:
% production
% death

kinit = k(:,1);
% beta = k(:,2);
% gamma = k(:,3);

Z = log(1./rand(nCells,1));
constant_rates = k(:,2:3) .* X;


[dt,a] = integral_search(magnitude_at_jumps,arrival_times,...
    interval_integrals,num_arrivals,t,nCells,kinit,constant_rates,Z,kappa,step);


% a(:,1) = a1;
% a(:,2) = beta .* X(:,1) .* dt;
% a(:,3) = gamma .* X(:,2).* dt;
t = t + dt;
a0 = sum(a,2);

% if ~any(isnan(sde_params))
%     k(:,1) = rate_update(sde_params,kinit,t,dt,nCells,nSteps);
% end
r2ao = a0.*rand(nCells,1);
mu = sum(repmat(r2ao,1,nRxn+1) >= cumsum([zeros(nCells,1),a],2),2);
if any(mu==0)
    disp('oh no');
end
return

function [tau,a] = integral_search(magnitude_at_jumps,arrival_times,...
    interval_integrals,num_arrivals,t,nCells,Kt,constant_rates,Z_,kappa,step)
% if step==1327
%     disp('check here!')
% end
Z = Z_;
t_ = t;
Kt_ = Kt;
c1 = sum(constant_rates,2);

arrival_times_ = arrival_times;
magnitude_at_jumps_ = magnitude_at_jumps;
k = sum(arrival_times <= t,2);
linind_last = sub2ind(size(arrival_times), (1:nCells)', k);
magnitude_at_jumps(linind_last) = Kt;
arrival_times(linind_last) = t;

not_last_arriv_ind = find(k<num_arrivals);
linind_next = sub2ind(size(arrival_times), not_last_arriv_ind, k(not_last_arriv_ind)+1);
interval_integrals(linind_next) = Kt(not_last_arriv_ind)/kappa ...
    .* (1-exp(-kappa*(arrival_times(linind_next) - t(not_last_arriv_ind))));

tau = zeros(nCells,1);
a = zeros(nCells,3);
G = zeros(nCells,1);

activecells = true(nCells,1);
% n_active_cells = nCells;
simindices = (1:nCells)';
while any(activecells)
    %this is an vector of length n_active_cells
    knew = k(activecells)+1;
    
    %find cells that have exceeded time of last jump
    %this is a boolean vector of length n_active_cells
    cells_past_last_jump = knew > num_arrivals(activecells);
    %these are indexing into the underying vector of length nCells
    cells_past_last_jump_ind = simindices(cells_past_last_jump);
    if any(cells_past_last_jump)
        C1 = c1(cells_past_last_jump_ind);
        C2 = Kt(cells_past_last_jump_ind)/kappa;
        C3 = C2 - Z(cells_past_last_jump_ind) ;
        
        d_tau = lambert_tau(C1,C2,C3,kappa);
        tau(cells_past_last_jump_ind) = tau(cells_past_last_jump_ind) + d_tau;
        dG_ = C2.*(1-exp(-kappa*d_tau)) + C1.*d_tau;
        G(cells_past_last_jump_ind) = G(cells_past_last_jump_ind) + dG_;
        Z(cells_past_last_jump_ind) = Z(cells_past_last_jump_ind) - dG_;
        
        activecells(cells_past_last_jump_ind) = false;
        if ~any(activecells)
            break
        end
    end
    knew = knew(~cells_past_last_jump);
    
    %identify cells that have not reached last jump
    
    %compute dt and dG at next jump for these cells
    simindices = find(activecells);
    linind_next = sub2ind(size(arrival_times), ...
        simindices, knew);
    t_curr  = t(activecells);
    t_next = arrival_times(linind_next);
    dt = t_next - t_curr;
    if any(dt<0)
        disp('?');
    end
    dG = interval_integrals(linind_next) + dt.*c1(activecells);
    
    %find cells that are before last jump that have terminated per rng
    term_cells = false(nCells,1);
    term_cells_subset =  dG>Z(activecells);
    term_cells(activecells) = term_cells_subset;
    if any(term_cells)
        C1 = c1(term_cells);
        C2 = Kt(term_cells)/kappa;
        C3 = C2 - Z(term_cells);
        d_tau = lambert_tau(C1,C2,C3,kappa);
        tau(term_cells) = tau(term_cells) + d_tau;
        dG_ = C2.*(1-exp(-kappa*d_tau)) + C1.*d_tau;
        G(term_cells) = G(term_cells) + dG_;
        Z(term_cells) = Z(term_cells) - dG_;
        
        activecells(term_cells) = false;
        if ~any(activecells)
            break
        end
    end
    k(activecells) = knew(~term_cells_subset);
    t(activecells) = t_next(~term_cells_subset);
    tau(activecells) = tau(activecells) + dt(~term_cells_subset);
    Kt(activecells) = magnitude_at_jumps(linind_next(~term_cells_subset));
    if ~isfinite(dG)
        disp('');
    end
    Z(activecells) = Z(activecells) - dG(~term_cells_subset);
    G(activecells) = G(activecells) + dG(~term_cells_subset);
    simindices = find(activecells);
end

a(:,2:3) = constant_rates.*tau;
a(:,1) = G-sum(a(:,2:3),2);
filt = ~isfinite(tau);
a(filt,1) = 1;
a(filt,2:3) = 0;
if tau==inf
    disp('');
end
% arrival_times==t_
% pause
if any(k>(num_arrivals+1))
    disp('oh nooo')
end
filt = a < -1e-6;
if any(filt(:))
    disp('oh no');
end
a(a<0) = 0;
t_new = t_+tau;
return

function W = iacono_boyd_lambert_high(logx,n)
% logx = log(x);
W = logx - log(1+logx/2);
for i = 1:n
    W = W./(1+W) .* (1 + logx - log(W));
end
return


function tau = lambert_tau(c1,c2,c3,kappa)
tau = NaN(size(c1));
logx = log(kappa*c2./c1) + kappa*c3./c1;
arg = exp(logx);
filt = arg<1000 & c1>0;
if any(filt) 
    tau(filt) = lambertw(arg(filt))/kappa - c3(filt)./c1(filt);
%     tau(filt) = iacono_boyd_lambert_low(arg(filt),logx(filt),5)/kappa - c3(filt)./c1(filt);
end
% tau = lambertw(0,arg)/kappa - c3./c1;
filt = arg>=1000 & c1 > 0;
if any(filt)
    tau(filt) = iacono_boyd_lambert_high(logx(filt),5)/kappa - c3(filt)./c1(filt);
end
% tau = Lambert_W(kappa*c2./c1.*exp(kappa*c3./c1),0)/kappa - c3./c1;

%check for cells with no linear term = i.e. where the only source of 
%reaction flux is the transcription reaction
filt = c1==0;
tau(filt) = -log(c3(filt)./c2(filt))/kappa;

%check for cells where the reaction flux is insufficient to yield a 
%reaction in finite time. This should ONLY occur when no molecules are
%present in the system.
filt = (filt & (c3./c2)<0);
% filt = (filt & (c3./c2)<0) | isnan(tau) | tau<0;
% filt = abs(imag(tau)) > 0 | tau < 0;
tau(filt)=inf;
if any(isnan(tau))
    disp('');
end
if any(abs(imag(tau))>0)
    disp('oh no!');
end
if any(tau<0)
    disp('oh no! x2');
end
return

function kinit = kinit_eval(sde_params, magnitude_at_jumps,arrival_times,t,nCells)
k = sde_params(1);
tind = sum(t>=arrival_times,2);
try
linindupdate = sub2ind(size(magnitude_at_jumps),(1:nCells)',tind);
catch
    disp('');
end
dt = t-arrival_times(linindupdate);
kinit = magnitude_at_jumps(linindupdate).*exp(-dt*k);
return

