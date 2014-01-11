function return_state = do_sparseLDA_v0(global_config)
return_state = 0;
% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx',
% 'train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

classification_config = global_config.classification;
splits = classification_config.splits;
descriptor_type = classification_config.descriptor_type;


% get which classes to classification
switch class(global_config.classification.class_idx)
    case 'double'
        do_class_idx = global_config.classification.class_idx;
        
    case 'char'
        do_class_idx = [1:class_num];
        
    otherwise
        fprintf('WRONG!!! wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    train_data = [];
    
    train_num_tot = 0;
    test_num_tot = 0;
    train_num_each_class = [];
    test_num_each_class = [];
    for i_class = 1:length(do_class_idx)
        cur_class = do_class_idx(i_class);
        
        train_num_each_class = [train_num_each_class,length( train_set_idx{cur_splits}{cur_class} )];
        test_num_each_class = [test_num_each_class,length( test_set_idx{cur_splits}{cur_class} )];
    end
    
    train_num_tot = sum(train_num_each_class);
    test_num_tot = sum(test_num_each_class);
    
    feat_num = 0;
    
    for i_desc = 1:length(descriptor_type)
        cur_descriptor = descriptor_type{i_desc};
        
        
        
        switch cur_descriptor
            case 'low_level'
                desc_path = global_config.extract_low_level_descriptors.path;
            case 'mid_level'
                desc_path = global_config.extract_mid_level_descriptors.path;
            otherwise
                fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                return_state = 0;
                return;
        end
        
        switch cur_descriptor
            case 'low_level'
                feature_types = classification_config.feature_types_low;
            case 'mid_level'
                feature_types = classification_config.feature_types_mid;
            otherwise
                fprintf('wrong descriptor type! \n');
                return;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            
            feat_num = feat_num +1;
            
            if length(classification_config.rho) > 1
                cur_rho = classification_config.rho(feat_num);
            else
                cur_rho = classification_config.rho(1);
            end
            
            load_name = fullfile(desc_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,1,1,cur_feature) );
            load(load_name);
            switch cur_descriptor
                case 'low_level'
                    descriptor = ll_descriptor;
                    
                case 'mid_level'
                    if (classification_config.rho >= 0)
                        ml_descriptor = 1./(1+exp(-cur_rho*ml_descriptor));
                    end
                    descriptor = ml_descriptor;
                    
                otherwise
                    fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                    return;
                    
            end
            
            cur_train_data = zeros( train_num_tot,length(descriptor) );
            cur_test_data = zeros( test_num_tot,length(descriptor) );
            
            i_row = 0;
            for i_class = 1:length(do_class_idx)
                cur_class = do_class_idx(i_class);
                fprintf('train loading split:%2d desc:%s feat:%s class:%3d\n',cur_splits,cur_feature,cur_feature,cur_class);
                
                vids_idx = train_set_idx{cur_splits}{cur_class};
                
                for i_vid = 1:length(vids_idx)
                    cur_vid = vids_idx(i_vid);
                    
                    
                    load_name = fullfile(desc_path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
                    load(load_name);
                    
                    
                    switch cur_descriptor
                        case 'low_level'
                            descriptor = ll_descriptor;
                            
                        case 'mid_level'
                            if (classification_config.rho >= 0)
                                ml_descriptor = 1./(1+exp(-cur_rho*ml_descriptor));
                            end
                            descriptor = ml_descriptor;
                            
                        otherwise
                            fprintf('WRONG!!! unknown video feature type! %s\n',classification_config.feature);
                            return;
                            
                    end
                    
                    % l2 normalize
                    descriptor = descriptor/norm(descriptor,2);
                    
                    i_row = i_row +1;
                    cur_train_data(i_row,:) = descriptor;
                    
                end
                
            end
            
            
            
            train_data = [train_data,cur_train_data(:,1:length(do_class_idx)*3)];
            
            
        end
        
    end
    
    train_label = zeros(size(train_data,1),length(do_class_idx));
    i_row = 0;
    for i_class = 1:length(do_class_idx)
        cur_class = do_class_idx(i_class);
        fprintf('train loading split:%2d desc:%s feat:%s class:%3d\n',cur_splits,cur_feature,cur_feature,cur_class);
        
        vids_idx = train_set_idx{cur_splits}{cur_class};
        
        train_label(i_row+1:i_row+length(vids_idx),i_class) = 1;
        
        i_row = i_row+length(vids_idx);
    end
    
    
    dim_num = size(train_data,2);
    weight = ones(dim_num,1);
    option = round(dim_num*classification_config.sparse_level);
    
    [BetaTab, ThetaTab, AlphaTab, lambda] = sparseLDA(train_data,train_label,weight,option);
    sparse_weight = sum(BetaTab{end}.^2,2) > 1e-6;
    sparse_weight = reshape(sparse_weight,feat_num,[]);
    sparse_weight = repmat(sparse_weight,1,24);
    
    sparse_base = cell(feat_num,1);
    feat_dim = size(BetaTab{end},1)/feat_num;
    for i_feat = 1:feat_num
        sparse_base{i_feat} = BetaTab{end}(feat_dim*(i_feat-1)+1:feat_dim*i_feat,:);       
    end
    
    % save
    save_name = fullfile(classification_config.path,sprintf('s%02d_sparse_weight.mat',cur_splits) );
    save(save_name,'BetaTab','ThetaTab','AlphaTab','lambda','sparse_weight','sparse_base');
    
end


end


%%
function [BetaTab, ThetaTab, AlphaTab, lambda] = sparseLDA(X,Y,weight,option,tol)
% [BETA, THETA, ALPHA, LAMBDA] = OSPENFITVAR1(X,Y,WEIGHT,OPTION,TOL) Optimal Scoring
%         fitting
%
% INPUTS:
%
% X      : (n,p) observation matrix
% Y      : (n,K) class indicator matrix
% WEIGHT : a penalty weight applied to each variable
% OPTION : if OPTION is scalar, it represents the maximum number of selected
%          variables, if OPTION is a vector, it represents the set of penalty
%          parameters.
% TOL    : Tolerance for the stopping criterion (magnitude of gradient)
%
% OUTPUTS:
%
% BETA   : structure containing the (p,K-1) matrix of regression parameters
% THETA  : structure containing the (K,K-1) matrix of optimal scores
% ALPHA  : structure containing the (1,K-1) eigenvalues in descending order
% LAMBDA : vector of penalty parameters

%  19/12/12 Y. Grandvalet & L. F. Sanchez Merchante
%  Bibliography
%
% @INPROCEEDINGS{Sanchez12,
%  author = {S\`anchez Merchante, L. F. and Grandvalet, Y. and Govaert, G.},
%  title = {An Efficient Approach to Sparse Linear Discriminant Analysis},
%  booktitle = {Proceedings of the 29th Annual International Conference on Machine
%               Learning (ICML 2012)},
%  year = {2012} }
%
% http://arxiv.org/abs/1206.6472
%
% revised 22/02/13 Y. Grandvalet (new THETA initialization)

if nargin < 5;
    tol = 1e-2;
    if nargin < 4;
        option = [];
        if nargin < 3;
            weight = [];
            if nargin < 2;
                error('OSPENFITVAR1 requires at least two input arguments.');
            end
        end
    end
end

% Check that matrix (X) and vector (Y) have compatible dimensions

[n p]  = size(X);
[ny K] = size(Y);
if ny~=n,
    error('The number of rows in Y must equal the number of rows in X.');
end

% Check that (weight) has correct dimensions

if isempty(weight),
    weight = ones(p,1);
    fprintf(1,'Running penalized optimal scoring with uniform weights.\r')
elseif length(weight) ~= p,
    error('WEIGHT should be a vector of size p.\r');
else
    fprintf(1,'Running penalized optimal scoring with non-uniform weights.\r')
end

% Check that (option) has correct dimensions

if isempty(option),
    maxvar = min(p,n) ; % 03/05/12 changed from maxvar = max(p,n)
    lambda = [];
    fprintf(1,'Running optimal scoring up to %d variables.\r', maxvar)
elseif length(option) == 1,
    if round(option)~=option,
        error('OPTION is scalar: should be an integer for the max. # of variables.\r');
    else
        maxvar = option;
        lambda = [];
        fprintf(1,'Running optimal scoring up to %d variables.\r', maxvar)
    end
else
    maxvar = p;
    lambda = option;
    n_lambda = length(lambda);
    fprintf(1,'Running optimal scoring with %d penalty parameter values.\r', n_lambda)
end

% Check that (tol) has correct dimensions

if length(tol) ~= 1,
    error('TOL must be a scalar.');
end

% Initializations

Beta = zeros(p,K-1);

Dpi = sum(Y,1)';
u = sqrt(Dpi/sum(Dpi));
[~,~,V] = svd(eye(K) - u*u');
Theta = diag(sqrt(n./Dpi))*V(:,1:K-1);

Y = Y*Theta;
Yhat = zeros(n,K-1);

J = 0.5*(sum(sum((Y-Yhat).^2)));
Pen = 0;
J_old = J ;

XX = zeros(p,p);
XY = X'*Y;

if isempty(lambda)
    lambda = (max(sqrt(sum(XY.^2,2))./weight)-2*tol)*(1-2*tol)*2.^[0:-0.5:-20];
    n_lambda = length(lambda);
end;

norm_min_subdiff = zeros(p,1);

ind_active = false(p,1) ; ilambda = 0 ; global_convergence=false;
while sum(ind_active)<maxvar && ilambda<n_lambda && ~global_convergence,
    ilambda=ilambda+1;
    fprintf('\n%d ',ilambda);
    iter = 0 ; convergence = false ;
    normB = sqrt(sum(Beta.^2,2));
    ind_active = (normB>0);
    while ~convergence;
        iter = iter + 1;
        fprintf('.');
        n_active = sum(ind_active);
        % SOLVE OS FOR THE CURRENT ACTIVE SET
        if n_active>0,
            val = realmax;
            while val>tol, %&& max(normB_active>tol),
                Beta = zeros(p,K-1) ;
                if min(normB(ind_active))<tol.^2, % Go for stablest resolution
                    Inv_sqrt_omega = sqrt(normB(ind_active)./weight(ind_active)) ;
                    C = chol(XX(ind_active,ind_active).*(Inv_sqrt_omega*Inv_sqrt_omega') + lambda(ilambda) * eye(n_active));
                    Beta(ind_active,:) = bsxfun(@times,Inv_sqrt_omega,C\(C'\bsxfun(@times,Inv_sqrt_omega,(XY(ind_active,:)))));
                    %disp('stable')
                else  % Go for cheapest resolution
                    Omega = weight(ind_active)./normB(ind_active) ;
                    C = chol(XX(ind_active,ind_active) + lambda(ilambda) * diag(Omega));
                    Beta(ind_active,:) = C\(C'\(XY(ind_active,:)));
                    %disp('cheap')
                    if any(isnan(Beta)); disp('NaN in Beta');%keyboard;
                    end;
                end;
                normB(ind_active) = sqrt(sum(Beta(ind_active,:).^2,2)) ; normB(ind_active); %keyboard;
                ind_nonzero = sqrt(sum(Beta.^2,2))>0 ;
                normB_nonzero = sqrt(sum(Beta(ind_nonzero,:).^2,2)) ;
                G = XX(ind_nonzero,ind_nonzero)*Beta(ind_nonzero,:)-XY(ind_nonzero,:);
                norm_min_subdiff  = sqrt(sum((G + bsxfun(@times,Beta(ind_nonzero,:),lambda(ilambda)*weight(ind_nonzero)./normB_nonzero)).^2,2));
                if any(isnan(norm_min_subdiff));disp('grad');disp('NaN in norm_min_subdiff') % keyboard;
                end;
                [val] = max(norm_min_subdiff);
            end
            %      disp(['grad ' val])
        end;
        convergence = true ;
        % IDENTIFY INACTIVATED GROUPS
        if min(normB(ind_active))<tol,
            ind_out = find(normB(ind_active)<tol);
            n_out = length(ind_out);
            i_out = 0;
            while convergence && i_out<n_out,
                i_out = i_out+1 ;
                coord = ind_out(i_out) ;
                Btmp = Beta(ind_active,:) ; Btmp(coord,:) = 0;
                tmp = find(ind_active); ind = tmp(coord);
                if sqrt(sum((XX(ind,ind_active)*Btmp-XY(ind,:)).^2,2))<(lambda(ilambda)*weight(ind));
                    ind_active(ind) = false;
                    Beta(ind,:) = 0;
                    convergence = false ;
                    % disp([num2str(ind) ' out'])
                end
            end
        end
        % IDENTIFY GREATEST VIOLATION OF OPTIMALITY CONDITIONS
        if convergence,
            G = XX(:,ind_active)*Beta(ind_active,:)-XY;
            normB = sqrt(sum(Beta.^2,2));
            norm_min_subdiff = zeros(p,1);
            norm_min_subdiff(~ind_active) = max(0,sqrt(sum(G(~ind_active,:).^2,2))-lambda(ilambda)*weight(~ind_active));
            [val, coord] = max(norm_min_subdiff);
            if val>tol,
                convergence = false ;
                % Adding features
                ind_active(coord) = true;
                XX(:,coord) = X'*X(:,coord);
                % block-coordinate update
                theta_coord = - G(coord,:)/sqrt(sum(G(coord,:).^2,2)) ;
                Beta_coord = XX(coord,coord)\(X(:,coord)'*Y-lambda(ilambda)*weight(coord)*theta_coord);
                Beta(coord,:) = Beta_coord;
                normB(coord) = sqrt(sum(Beta_coord.^2,2));
                % disp([num2str(coord) ' in'])
            end
        end
        %   convergence monitoring
        if rem(iter,100)==0,
            Yhat = X(:,ind_active)*Beta(ind_active,:);
            RSS = 0.5*sum(sum((Y-Yhat).^2));
            Pen = lambda(ilambda)*sum(weight(ind_active).*sqrt(sum(Beta(ind_active,:).^2,2)));
            J = RSS + Pen;
            %fprintf(1,'iter: %d, obj: %1.4f, RSS: %1.4f, Penalty: %1.4f #0: %d CV: %1.4f \r',iter, RSS+Pen,RSS,Pen,p-sum(ind_active),(J_old-J)/tol.^2)
        end
        J_old = J;
    end
    Yhat = X(:,ind_active)*Beta(ind_active,:);
    RSS = 0.5*sum(sum((Y-Yhat).^2));
    Pen = lambda(ilambda)*sum(weight(ind_active).*sqrt(sum(Beta(ind_active,:).^2,2)));
    J = RSS + Pen;
    %fprintf(1,'iter: %d, obj: %1.4f, RSS: %1.4f, Penalty: %1.4f #0: %d CV: %1.4f \r',iter, RSS+Pen,RSS,Pen,p-sum(ind_active),(J_old-J)/tol.^2)
    % Optimal scores
    [~,S,V] = svd(Y'*Yhat);
    BetaTab{ilambda}  = Beta*V;
    ThetaTab{ilambda} = Theta*V;
    alpha = sqrt(diag(S)/n);
    AlphaTab{ilambda} = alpha;
    if max(sqrt(sum(G.^2,2)))<tol,
        disp('Up to TOL, the final solution is the OLS solution');
        global_convergence=true;
    else
        % fprintf(1,'Penalty parameter # %d \r',ilambda)
    end
end
lambda = lambda(1:ilambda);

end