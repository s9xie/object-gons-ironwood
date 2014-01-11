% wyw 9/6/2013 @MSRA
% input: codes, for FV coding, the codes is generate just before used
% output: low level descriptors of videos s%02d_c%03d_v%03d_[feature_type].mat;

% v0 -> v1: support Fisher Vector coding before extract low level
% descriptors

function return_state = m_extract_low_level_descriptors_v1(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

elld_config = global_config.extract_low_level_descriptors;
splits = elld_config.splits;
feature_types = elld_config.feature_types;


% specify the videos to do
do_sets = global_config.extract_low_level_descriptors.class_idx;

do_vids = cell(length(splits),class_num);
switch class(do_sets)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets{i}(1)} = [do_vids{cur_splits,do_sets{i}(1)};do_sets{i}(2)];
            end
            
        end
    case 'double'
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets(i)} = [train_set_idx{cur_splits}{do_sets(i)};test_set_idx{cur_splits}{do_sets(i)}];
            end
        end
        
    case 'char'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:class_num
                do_vids{cur_splits,i} = [train_set_idx{cur_splits}{i};test_set_idx{cur_splits}{i}];
            end
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

if strcmp(global_config.coding.params.method,'FV')
    vl_setup;
end

FV_vars = [];

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    for i_class = 1:class_num
        
        cur_class = i_class;
        if isempty(do_vids{cur_splits,i_class})
            continue;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            fprintf('extract low level split:%d class:%d feature:%s\n',cur_splits,cur_class,cur_feature);
            % all data
            cur_vid_idx = do_vids{cur_splits,i_class};
            
            if strcmp(global_config.coding.params.method,'FV')
                % load the related GMM model
                load_name = fullfile(global_config.learn_codebooks.path,sprintf('gmms_s%02d_%s.mat',cur_splits,cur_feature));
                load(load_name,'gmm_means','gmm_covs','gmm_priors','pca_base','pca_mean');
                FV_vars.gmm_means = gmm_means;
                FV_vars.gmm_covs = gmm_covs ;
                FV_vars.gmm_priors = gmm_priors;
                FV_vars.pca_base = pca_base;
                FV_vars.pca_mean = pca_mean;
            end
            
            switch elld_config.params.method
                case 'max'
                    % do_pooling_max
                     if global_config.num_core > 1
                        parfor i_vid = 1:length(cur_vid_idx)
                            do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),FV_vars,global_config);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),FV_vars,global_config);
                        end
                    end
                                       
                otherwise
                    fprintf('wrong type of elld_config.params.method %s!\n',elld_config.params.method);
                    return_state = 0;
                    return;
            end
            
        end  % ifeature_type
    end % iclass ends
    
end % i_split

end

%%
function do_pooling_max(cur_splits,cur_feature,cur_class,cur_vid,FV_vars,global_config)

elld_config = global_config.extract_low_level_descriptors;
codes_path = global_config.coding.path;
feature_path = global_config.extract_features.path;



switch global_config.coding.params.method
    case 'LSAQ' % LSAQ has the codes coded in the section coding, now just load the codes stored
        load_name = fullfile(codes_path,sprintf('codes_s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature));
        load(load_name,'codes');
    case 'FV' % no storage, need to do coding here
        
       
        gmm_means = FV_vars.gmm_means;
        gmm_covs = FV_vars.gmm_covs;
        gmm_priors = FV_vars.gmm_priors;
        pca_base = FV_vars.pca_base;
        pca_mean = FV_vars.pca_mean;
        
        
        coding_config = global_config.coding;
                
        % load the feature
        load_name = fullfile(feature_path,sprintf('c%03d_v%03d_features_%s.mat',cur_class,cur_vid,cur_feature) );
        load(load_name);
        
        switch cur_feature
            case 'shape'
                features = features_shape;
            case 'hog'
                features = features_hog;
            case 'hof'
                features = features_hof;
            case 'mbhx'
                features = features_mbhx;
            case 'mbhy'
                features = features_mbhy;
            otherwise
                fprintf('wrong feature type %s\n',cur_feature);
                return;
        end
        
        fprintf('extract low level descriptor split:%d class:%d vid:%d feature:%s dt_num:%d\n',cur_splits,cur_class,cur_vid,cur_feature,size(features,1));
        
        % perform PCA
        features = features - repmat(pca_mean,size(features,1),1);
        features = features*pca_base;
        
        % FV coding using vlfeat
        codes = zeros(size(features,1),2*size(features,2)*length(gmm_priors));
        
        if global_config.num_core > 1
            parfor i = 1:size(features,1)
                codes(i,:) = vl_fisher(features(i,:)', gmm_means,gmm_covs,gmm_priors,...
                    'Normalized','SquareRoot')';
            end
        else
            for i = 1:size(features,1)
                cur_code = vl_fisher(features(i,:)', gmm_means,gmm_covs,gmm_priors)';
                % l1 norm;
                cur_code = cur_code/sum(abs(cur_code));
                
                % element wise root square
                cur_code = sign(cur_code).*sqrt(abs(cur_code));
                
                
                codes(i,:) = cur_code;
                
            end
        end
        if sum(sum(isnan(codes)))
            fprintf('NaN data occur!\n');
            return;
        end
    otherwise
        fpritnf('WRONG types of coding methods!\n');
        return;
end

load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
load(load_name,'vid_info');

load_name = fullfile(feature_path,sprintf('c%03d_v%03d_location.mat',cur_class,cur_vid));
load(load_name,'location');

switch elld_config.params.pooling_range_type
    
    case 'video'
        pooling_range = [vid_info(1),vid_info(1),vid_info(2),vid_info(2),vid_info(3),vid_info(3)];
        pooling_cells = elld_config.params.pooling_cells.*...
    (repmat(pooling_range,size(elld_config.params.pooling_cells,1),1));
        
    case 'bbox'
        min_loc = min(location(:,1:3),[],1);
        delta_loc = max(location(:,1:3),[],1) - min_loc;
        
        low_range = [min_loc(2),min_loc(2),min_loc(3),min_loc(3),min_loc(1),min_loc(1)];
        delta_range = [delta_loc(2),delta_loc(2),delta_loc(3),delta_loc(3),delta_loc(1),delta_loc(1)];
        
        pooling_cells = repmat(low_range,size(elld_config.params.pooling_cells,1),1) + ...
            elld_config.params.pooling_cells.* repmat(delta_range,size(elld_config.params.pooling_cells,1),1);     
        
    otherwise
            fprintf('WRONG!!! the type of pooling range %s not exist!\n ',elld_config.params.pooling_range);
    
end


pooling_cells_num = length(pooling_cells);


% pooling --------------------
ll_descriptor = [];
for i_cell = 1:pooling_cells_num
    
    select_id = (location(:,2) >= pooling_cells(i_cell,1))&(location(:,2) <= pooling_cells(i_cell,2))&...
        (location(:,3) >= pooling_cells(i_cell,3))&(location(:,3) <= pooling_cells(i_cell,4))&...
        (location(:,1) >= pooling_cells(i_cell,5))&(location(:,1) <= pooling_cells(i_cell,6));
    
    codes_in_cell = codes(find(select_id == 1),:);
    if(size(codes_in_cell,1) > 0 )
        ll_descriptor = [ll_descriptor,full( max(codes_in_cell,[],1))];
    else
        ll_descriptor = [ll_descriptor,zeros(1,size(codes_in_cell,2))];
    end
end

if sum(sum(isnan(ll_descriptor))) 
    printf('WRONG!! NaN occur!\n');
end

save_name = fullfile(elld_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat', cur_splits,cur_class,cur_vid,cur_feature) );
save(save_name,'ll_descriptor');

% clear codes;

end

%%
function codes = do_coding_FV(cur_splits,cur_feature,cur_class,cur_vid,global_config)


% load the related GMM model
load_name = fullfile(global_config.learn_codebooks.path,sprintf('gmms_s%02d_%s.mat',cur_splits,cur_feature));
load(load_name,'gmm_means','gmm_covs','gmm_priors','pca_base','pca_mean');

feature_path = global_config.extract_features.path;
coding_config = global_config.coding;


% load the feature
load_name = fullfile(feature_path,sprintf('c%03d_v%03d_features_%s.mat',cur_class,cur_vid,cur_feature) );
load(load_name);

switch cur_feature
    case 'shape'
        features = features_shape;
    case 'hog'
        features = features_hog;
    case 'hof'
        features = features_hof;
    case 'mbhx'
        features = features_mbhx;
    case 'mbhy'
        features = features_mbhy;
    otherwise
        fprintf('wrong feature type %s\n',cur_feature);
        return;
end

% perform PCA
features = features - repmat(pca_mean,size(features,1),1);
features = features*pca_base;

% FV coding using vlfeat
codes = zeros(size(features,1),2*size(features,2)*length(gmm_priors));

if global_config.num_core > 1
    parfor i = 1:size(features,1)
        codes(i,:) = vl_fisher(features(i,:)', gmm_means,gmm_covs,gmm_priors,...
            'Normalized','SquareRoot')';
    end
else
    for i = 1:size(features,1)
        codes(i,:) = vl_fisher(features(i,:)', gmm_means,gmm_covs,gmm_priors)';
		% L1 norm
		
		% square root each dimension
		
		
    end
end
if sum(sum(isnan(codes)))
   fprintf('NaN data occur!\n');
   return;
end

% codes = sparse(codes);

% save_name = fullfile(coding_config.path,sprintf('codes_s%02d_c%03d_v%03d_%s.mat',cur_splits,i_class,i_vid,cur_feature));
% save(save_name,'codes','-v7.3');

end

