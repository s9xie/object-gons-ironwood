%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 9/6/2013
% wyw @MSRA


% call the dense trajectory feature to get video dense trajectory features,
% input: dataset_info.mat
% output: videos feature mat @global_config.extract_features.path
%               c%03d_v%03d_info.mat: [width, height, len, dt_number]
%               c%03d_v%03d_location.mat [x,y,t,....]
%               c%03d_v%03d_trajectory.mat [x_1,y_1,x_2,y_2,...,x_dt_len+1,y_dt_len+1]
%               c%03d_v%03d_features_shape.mat
%               c%03d_v%03d_features_hog.mat
%               c%03d_v%03d_features_hof.mat
%               c%03d_v%03d_features_mbhx.mat
%               c%03d_v%03d_features_mbhy.mat
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% v1:support bbox file

function return_state = m_extract_features_v1(global_config)
return_state =1;
% load data set info
% 'vid_paths','vid_names',vid_bbox_names,'class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

% extract feature
tic;

do_sets = global_config.extract_features.class_idx;
do_vids = cell(class_num,1); % each cell specifys the videos idx in the related class

switch class(do_sets)
    case 'cell' 
        for i = 1:length(do_sets)
            do_vids{do_sets{i}(1)} = [do_vids{do_sets{i}(1)};do_sets{i}(2)];
        end
    case 'double'
        for i = 1:length(do_sets)
            do_vids{do_sets(i)} = 1:vid_nums_in_class(do_sets(i));
        end
        
    case 'char'
        for i = 1:class_num
            do_vids{i} = 1:vid_nums_in_class(i);
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end



for i_class = 1:class_num
    cur_class = i_class;
    
    if isempty(do_vids{cur_class})
        continue;
    end
    
    cur_vids_idx = do_vids{cur_class};
    
    if global_config.num_core > 1
        parfor i_vid = 1: length(cur_vids_idx)
           do_extract_features_v1(cur_class,cur_vids_idx(i_vid),vid_paths,vid_bbox_names,global_config);
        end
    else
        for i_vid = 1: length(cur_vids_idx)
            do_extract_features_v1(cur_class,cur_vids_idx(i_vid),vid_paths,vid_bbox_names,global_config);           
        end
    end
    toc;
end
        
  

end

%%
% wyw 9/6/2013 @MSRA
% call extract_fea_fun to extract and save features
% graph cut all the trajectories, get the trajectory clusters
% input: trajectory and location
% output: clusters_c%03d_v%03d.mat every row is an cluster reslults, and
% the location of the center of the cluster

% v1:support bbox file

function do_extract_features_v1(cur_class,cur_vid,vid_paths,vid_bbox_names,global_config)

extract_fea_fun = global_config.extract_features.extract_fea_fun;
video_name = fullfile(global_config.dataset_path,vid_paths{cur_class}{cur_vid});
vid_bbox_name = fullfile([global_config.dataset_path,'_bbox'],vid_bbox_names{cur_class}{cur_vid});
min_dt_num = global_config.extract_features.min_dt_num;
max_dt_num = global_config.extract_features.max_dt_num;
feature_path = global_config.extract_features.path;
i_class = cur_class;
i_vid = cur_vid;

% extract_fea_fun,video_name,vid_bbox_name,min_dt_num,max_dt_num,i_class,i_vid,feature_path
%  do_extract_features_v1(global_config.extract_features.extract_fea_fun,...
%                 fullfile(global_config.dataset_path,vid_paths{cur_class}{cur_vids_idx(i_vid)}),...
%                fullfile([global_config.dataset_path,'_bbox'],vid_bbox_names{cur_class}{cur_vids_idx(i_vid)}),...
%                global_config.extract_features.min_dt_num,...
%               global_config.extract_features.max_dt_num,...
%             cur_class,cur_vids_idx(i_vid),global_config.extract_features.path );

if ~exist(extract_fea_fun,'file')
    fprintf('%s did not exist! \n',extract_fea_fun);
    return;
end

fprintf('class:%2d file:%3d ',i_class,i_vid);

temp_feature_name = sprintf('temp_feature_c%03d_v%03d',i_class,i_vid);
temp_vid_info_name = sprintf('temp_vid_info_c%03d_v%03d',i_class,i_vid);
if ~exist(vid_bbox_name,'file')
    vid_bbox_name = 'no_bbox';
    fprintf(' no_bbox ');
end


% call exe to get temp file
cmd_str = [extract_fea_fun,' ',video_name,' ',vid_bbox_name,' ',num2str(min_dt_num),' ',num2str(max_dt_num),' ',temp_feature_name,' ',temp_vid_info_name];
dos(cmd_str);

% read from temp file
if ~(exist(temp_feature_name,'file') && exist(temp_vid_info_name,'file'))
    fprintf('dt feature extract wrong!\n');
    return;
end

[dt_features,vid_info] = read_dt_data_mex(temp_feature_name,temp_vid_info_name);
% delete temp file
delete(temp_feature_name);
delete(temp_vid_info_name);


dt_features = dt_features';


if sum(sum(isnan(dt_features)))
    fprintf('wrong!!!!!!!!!!!!! %s \n',video_name);
    
    valid_idx = find(sum(isnan(dt_features),2)==0);
    
    if size(dt_features,1) - length(valid_idx) < 10
        dt_features = dt_features(valid_idx,:);
        fprintf('remove invalid dt features, continue\n');
        
    else
        return;
    end
    
    
else
    fprintf('dt_num:%d %s \n',size(dt_features,1),video_name);
end

% save features
% [width, height, len, location_dim, trajectory_dim, shape_dim, hog_dim, hof_dim, mbhx_dim, mbhy_dim]
fea_dim = vid_info(4:end);
fea_dim = cumsum(fea_dim);

% [width, height, len, dt_number]
vid_info = vid_info(1:3);
vid_info = [vid_info;size(dt_features,1)];

save_name = sprintf('%s\\c%03d_v%03d_info.mat',feature_path,i_class,i_vid);
save(save_name,'vid_info');

location = dt_features(:,1:fea_dim(1));
save_name = sprintf('%s\\c%03d_v%03d_location.mat',feature_path,i_class,i_vid);
save(save_name,'location');

trajectory = dt_features(:,fea_dim(1)+1:fea_dim(2));
save_name = sprintf('%s\\c%03d_v%03d_trajectory.mat',feature_path,i_class,i_vid);
save(save_name,'trajectory');

features = cell(length(fea_dim)-2,1);
features_shape = dt_features(:,fea_dim(2)+1:fea_dim(3));
features_hog = dt_features(:,fea_dim(3)+1:fea_dim(4));
features_hof = dt_features(:,fea_dim(4)+1:fea_dim(5));
features_mbhx = dt_features(:,fea_dim(5)+1:fea_dim(6));
features_mbhy = dt_features(:,fea_dim(6)+1:fea_dim(7));


save_name = sprintf('%s\\c%03d_v%03d_features_shape.mat',feature_path,i_class,i_vid);
save(save_name,'features_shape');

save_name = sprintf('%s\\c%03d_v%03d_features_hog.mat',feature_path,i_class,i_vid);
save(save_name,'features_hog');

save_name = sprintf('%s\\c%03d_v%03d_features_hof.mat',feature_path,i_class,i_vid);
save(save_name,'features_hof');

save_name = sprintf('%s\\c%03d_v%03d_features_mbhx.mat',feature_path,i_class,i_vid);
save(save_name,'features_mbhx');

save_name = sprintf('%s\\c%03d_v%03d_features_mbhy.mat',feature_path,i_class,i_vid);
save(save_name,'features_mbhy');

end