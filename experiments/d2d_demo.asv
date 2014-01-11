close all;
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add related path
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% addpath ..\common\m_functions\;     % path for matlab m functions
% addpath ..\common\mex_functions\;   % path for mex functions
% addpath ..\common\exe_functions\;   % path for exe functions
% addpath ..\tools\;  
% addpath ..\tools\vlfeat-0.9.17\toolbox\; % vlfeat toolbox
% addpath ..\tools\libsvm-3.17\matlab\; 
% addpath ..\tools\liblinear-1.93\matlab\;
% addpath ..\tools\m5il\; % solving CCCP
% addpath ..\tools\curve_toolbox; % show curves
% addpath ..\tools\curve_toolbox\ConfusionMatrices;

%% global_config of experiment seting

global_config.dataset = 'Scene-15'; % data set name
% set core num %%%%%%%%%%%%%%%%%%%%%%%%%%
global_config.num_core = 1; % nunber core of matlabpool

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global_config.load_visual_concepts = 1;

global_config.load_visual_concepts.func = 'm_load_visual_concepts_v1';
global_config.vc_path = 'C:\Users\s9xie\Desktop\Workspace\concepts_detection\Experiments\15_scene\miSVM_cltr_716_cat20_ori_July.mat\feat_data\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract feature part
global_config.extract_features.do = 1;

global_config.extract_features.version = '1';
global_config.extract_features.func = 'm_extract_features_v1'; % the m function to used
global_config.extract_features.extract_fea_fun = '..\common\exe_functions\get_dt_feature_refine.exe'; % specify the function to extract dense trajectory feature

global_config.extract_features.class_idx = [1:51];  % spesify which class and videos to extract feature
global_config.extract_features.min_dt_num = 500; % min number of  dense trajectories extracted from a video
global_config.extract_features.max_dt_num = 500000; % max number of dense trajectories extracted from a video
global_config.extract_features.feature_types = {'shape','hog','hof','mbhx','mbhy'};
global_config.extract_features.params.dt_len = 15; % the length of dense trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% learn codebooks part
global_config.learn_codebooks.do = 1;
global_config.learn_codebooks.func = 'm_learn_codebooks_v1';
global_config.learn_codebooks.version = ...
	[global_config.extract_features.version,'1'];

global_config.learn_codebooks.splits = [1]; % spesify the splits of experiment
global_config.learn_codebooks.feature_types = {'hof','hog','shape','mbhx','mbhy'};

global_config.learn_codebooks.method = 'FV'; % 'BoW' or 'FV', learn the codebook of bag of words or the GMM model of fisher vector

global_config.learn_codebooks.params.BoW.codebook_size = 4000;
global_config.learn_codebooks.params.BoW.select_data_num_total = 100000;
global_config.learn_codebooks.params.BoW.select_data_num_each_class = 2000;

global_config.learn_codebooks.params.FV.K = 256;
global_config.learn_codebooks.params.FV.reduced_factor = 0.5; % the factor of dim reduce
global_config.learn_codebooks.params.FV.not_reduced_max_dim = 64; % the max dim not to reduce
global_config.learn_codebooks.params.FV.select_data_num_total = 256000;
global_config.learn_codebooks.params.FV.select_data_num_each_class = 256000/50;

global_config.learn_codebooks.params.re_select_data = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% coding
global_config.coding.do = 1;
global_config.coding.func = 'm_coding_v1';

global_config.coding.version = ...
    [global_config.learn_codebooks.version,'1'];

global_config.coding.params.method = 'FV'; %'LSAQ' or 'FV'. For FV the size of codes is too large to store, so put coding into extract_low_level_descriptors and extract_parts_features
global_config.coding.splits = [1]; 
global_config.coding.class_idx = [1:51];  % spesify which class and videos to extract feature
global_config.coding.feature_types = {'shape','hog','hof','mbhx','mbhy'}; 

global_config.coding.params.KNN = 5;
global_config.coding.params.beta = -10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract low_level feature
global_config.extract_low_level_descriptors.do = 1;
global_config.extract_low_level_descriptors.func = 'm_extract_low_level_descriptors_v1';
global_config.extract_low_level_descriptors.version = ...
    [global_config.coding.version,'1'];

global_config.extract_low_level_descriptors.splits = [1];
global_config.extract_low_level_descriptors.class_idx = [1:51];  % spesify which class and videos to extract feature
global_config.extract_low_level_descriptors.feature_types = {'shape','hog','hof','mbhx','mbhy'};  % spesify which class and videos to extract feature

global_config.extract_low_level_descriptors.params.pooling_range_type = 'video';  % video or bbox, bbox is the bounding box of all the trajectories
global_config.extract_low_level_descriptors.params.method = 'max';
global_config.extract_low_level_descriptors.params.pooling_cells = [
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0;

	0.0, 1.0, 0.0, 1.0/3.0, 0.0, 1.0;
	0.0, 1.0, 1.0/3.0, 2.0/3.0, 0.0, 1.0;
	0.0, 1.0, 2.0/3.0, 1.0, 0.0, 1.0;

	0.0, 1.0/2.0, 0.0, 1.0/2.0, 0.0, 1.0;
	0.0, 1.0/2.0, 1.0/2.0, 1.0, 0.0, 1.0;
	1.0/2.0, 1.0, 0.0, 1.0/2.0, 0.0, 1.0;
	1.0/2.0, 1.0, 1.0/2.0, 1.0, 0.0, 1.0;

	0.0, 1.0, 0.0, 1.0, 0.0, 1.0/2.0;
	0.0, 1.0, 0.0, 1.0, 1.0/2.0, 1.0;

	0.0, 1.0, 0.0, 1.0/3.0, 0.0, 1.0/2.0;
	0.0, 1.0, 1.0/3.0, 2.0/3.0, 0.0, 1.0/2.0;
	0.0, 1.0, 2.0/3.0, 1.0, 0.0, 1.0/2.0;
	0.0, 1.0, 0.0, 1.0/3.0, 1.0/2.0, 1.0;
	0.0, 1.0, 1.0/3.0, 2.0/3.0, 1.0/2.0, 1.0;
	0.0, 1.0, 2.0/3.0, 1.0, 1.0/2.0, 1.0;

	0.0, 1.0/2.0, 0.0, 1.0/2.0, 0.0, 1.0/2.0;
	0.0, 1.0/2.0, 1.0/2.0, 1.0, 0.0, 1.0/2.0;
	1.0/2.0, 1.0, 0.0, 1.0/2.0, 0.0, 1.0/2.0;
	1.0/2.0, 1.0, 1.0/2.0, 1.0, 0.0, 1.0/2.0;
	0.0, 1.0/2.0, 0.0, 1.0/2.0, 1.0/2.0, 1.0;
	0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0/2.0, 1.0;
	1.0/2.0, 1.0, 0.0, 1.0/2.0, 1.0/2.0, 1.0;
	1.0/2.0, 1.0, 1.0/2.0, 1.0, 1.0/2.0, 1.0;
];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read data set info part
global_config.read_dataset_info.do = 1; % read data set info

global_config.read_dataset_info.func = 'm_read_dataset_info_v2';
global_config.read_dataset_info.version = '1'; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cluster_visual_concept
global_config.vc_clustering.do = 1;
global_config.vc_clustering.func = 'm_vc_clustering_v2';
global_config.vc_clustering.version = [global_config.extract_features.version,'1'];

global_config.vc_clustering.class_idx = [1:15];  % spesify which class and videos to extract feature

global_config.vc_clustering.params.min_cluster_size = 50;
global_config.vc_clustering.params.target_cluster_num = 100;

global_config.vc_clustering.params.method = 'GANC'; % method for cluster

global_config.vc_clustering.params.min_cluster_num = 10;

global_config.vc_clustering.params.co_exist_len = 5;
global_config.vc_clustering.params.NN = 20;
global_config.vc_clustering.params.spatial_dis_thread = 20;
global_config.vc_clustering.params.exp_beta = -0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cluster
global_config.clustering.do = 1;
global_config.clustering.func = 'm_clustering_v2';
global_config.clustering.version = ...
    [global_config.extract_features.version,'1'];

global_config.clustering.class_idx = [1:51];  % spesify which class and videos to extract feature

global_config.clustering.params.min_cluster_size = 50;
global_config.clustering.params.target_cluster_num = 100;

global_config.clustering.params.method = 'GANC'; % method for cluster

global_config.clustering.params.min_cluster_num = 10;

global_config.clustering.params.co_exist_len = 5;
global_config.clustering.params.NN = 20;
global_config.clustering.params.spatial_dis_thread = 20;
global_config.clustering.params.exp_beta = -0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract_part_features
global_config.extract_part_features.do = 1;
global_config.extract_part_features.func = 'm_extract_part_features_v3_vc';
global_config.extract_part_features.version = [global_config.clustering.version,'1'];


%global_config.extract_part_features.feature_types = {'pair','shape','hog','hof','mbhx','mbhy'}; 
%global_config.extract_part_features.feature_types = {'pair'}; 
global_config.extract_part_features.feature_types = {''}; 

global_config.extract_part_features.params.method = 'cluster'; % cluster | VOI
global_config.extract_part_features.params.pooling = 'max';

global_config.extract_part_features.splits = [1]; % learn codebook of splits 1,2,3
global_config.extract_part_features.class_idx = [1:15];  % spesify which class and videos to extract feature

global_config.extract_part_features.pair_params.time_mu = [0,5,10,20,40,80,120];
global_config.extract_part_features.pair_params.time_var = [1,1,1,1,1,1,1];
global_config.extract_part_features.pair_params.location_mu = [0.05,0.1,0.2,0.4,0.6,0.8];
global_config.extract_part_features.pair_params.location_var = 1./[10,10,10,10,10,10];
global_config.extract_part_features.pair_params.motion_mu = [0,0.05,0.1,0.15,0.2,0.25];
global_config.extract_part_features.pair_params.motion_var = 1./[100,100,100,100,100,100];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% learn_action_parts
global_config.learn_action_parts.do = 1;

global_config.learn_action_parts.struct_M5IL.graph.node_num = 2;
global_config.learn_action_parts.version = ...
    [global_config.extract_part_features.version,'1'];

if global_config.learn_action_parts.struct_M5IL.graph.node_num == 1
    global_config.learn_action_parts.func = 'm_learn_action_parts_v0_vc';
else
    global_config.learn_action_parts.func = 'm_learn_action_parts_v2_vc';
end


global_config.learn_action_parts.splits = [1]; % learn codebook of splits 1,2,3
global_config.learn_action_parts.class_idx = [1:15];  % spesify which class and videos to extract feature
%global_config.learn_action_parts.feature_types = {'hof'}; 
global_config.learn_action_parts.feature_types = {'vc'}; 
global_config.learn_action_parts.init.using_rand = 1; % add rand to init W

if global_config.learn_action_parts.struct_M5IL.graph.node_num == 3
	global_config.learn_action_parts.struct_M5IL.method = 'TRW-S'; % for node number is 3
else
	global_config.learn_action_parts.struct_M5IL.method = 'traversal';
end

global_config.learn_action_parts.reload = 1;

% ----------------- struct_M5IL verison parameters -----------------
global_config.learn_action_parts.struct_M5IL.using_parallel = false;
if global_config.num_core > 1
    global_config.learn_action_parts.struct_M5IL.using_parallel = true;
end
                        
global_config.learn_action_parts.struct_M5IL.channel_num = 3;

global_config.learn_action_parts.struct_M5IL.init.rand_init_num = 5;

global_config.learn_action_parts.struct_M5IL.init.liblinear.C = 1;
global_config.learn_action_parts.struct_M5IL.init.liblinear.bias = 1;                     

global_config.learn_action_parts.struct_M5IL.stop.criterion = {'numIterCCCP', 'detaLoss', 'detaLossRate'};
global_config.learn_action_parts.struct_M5IL.stop.maxNumIterCCCP = 5;
global_config.learn_action_parts.struct_M5IL.stop.minDetaLoss = .1;
global_config.learn_action_parts.struct_M5IL.stop.minDetaLossRate = .01;

global_config.learn_action_parts.struct_M5IL.C_neg = 2^8;
global_config.learn_action_parts.struct_M5IL.C_pos = 2^10;
global_config.learn_action_parts.struct_M5IL.C_div = 2^5;
global_config.learn_action_parts.struct_M5IL.zeta = 10;

global_config.learn_action_parts.struct_M5IL.eps = .02;
global_config.learn_action_parts.struct_M5IL.siz_incremt_ws = 100;

global_config.learn_action_parts.struct_M5IL.solveQP.init_from_last = true;      % 
global_config.learn_action_parts.struct_M5IL.solveQP.solver = 'matlab';          % 'matlab', 'svm-struct';


% ----------------- M5IL verison parameters -----------------

global_config.learn_action_parts.M5IL.channel_num = 3;
global_config.learn_action_parts.M5IL.init.rand_init_num = 2;
global_config.learn_action_parts.M5IL.init.liblinear.C = 1;
global_config.learn_action_parts.M5IL.init.liblinear.bias = 1;

global_config.learn_action_parts.M5IL.init.type = 'spec'; %'auto';
global_config.learn_action_parts.M5IL.init.ind_spec = 1;
global_config.learn_action_parts.M5IL.init.wInit = [];                           

global_config.learn_action_parts.M5IL.stop.criterion = {'numIterCCCP', 'detaLoss', 'detaLossRate'};
global_config.learn_action_parts.M5IL.stop.maxNumIterCCCP = 5;
global_config.learn_action_parts.M5IL.stop.minDetaLoss = .1;
global_config.learn_action_parts.M5IL.stop.minDetaLossRate = .01;

global_config.learn_action_parts.M5IL.alpha = 2^10;
global_config.learn_action_parts.M5IL.beta = 2^12;
global_config.learn_action_parts.M5IL.gamma = 2^5;
global_config.learn_action_parts.M5IL.zeta = 10;

global_config.learn_action_parts.M5IL.eps = .02;
global_config.learn_action_parts.M5IL.siz_incremt_ws = 100;

global_config.learn_action_parts.M5IL.solveQP.init_from_last = true;      % 
global_config.learn_action_parts.M5IL.solveQP.solver = 'matlab';          % 'matlab', 'svm-struct';

global_config.learn_action_parts.M5IL.compt_loss_cccp = true;
global_config.learn_action_parts.M5IL.compt_loss zh_ub = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract_mid_level_descriptors
global_config.extract_mid_level_descriptors.do = 1;

global_config.extract_mid_level_descriptors.func = 'm_extract_mid_level_descriptors_v3_vc';
global_config.extract_mid_level_descriptors.version = ...
    global_config.learn_action_parts.version;

global_config.extract_mid_level_descriptors.splits = [1];
global_config.extract_mid_level_descriptors.class_idx = [1:15];  % spesify which class and videos to extract feature
global_config.extract_mid_level_descriptors.feature_types = ... % *-X: X means the number of nodes in graph, i.e. -1 is one-gons, -2 is two-gons, -3 is three gons
    {'vc-2'
%         'hof-1','hog-1','shape-1','mbhx-1','mbhy-1',...
%         'hof-2','hog-2','shape-2','mbhx-2','mbhy-2',...
%    'hof-3','hog-3','shape-3','mbhx-3','mbhy-3',...
    };  % spesify which class and videos to extract feature


global_config.extract_mid_level_descriptors.params.method = 'max';
global_config.extract_mid_level_descriptors.params.pooling_range_type = 'video';  % video or bbox
global_config.extract_mid_level_descriptors.params.pooling_cells = [
    0.0, 1.0, 0.0, 1.0;

	0.0, 1.0, 0.0, 1.0/3.0;
	0.0, 1.0, 1.0/3.0, 2.0/3.0;
	0.0, 1.0, 2.0/3.0, 1.0;

	0.0, 1.0/2.0, 0.0, 1.0/2.0;
	0.0, 1.0/2.0, 1.0/2.0, 1.0;
	1.0/2.0, 1.0, 0.0, 1.0/2.0;
	1.0/2.0, 1.0, 1.0/2.0, 1.0;

	0.0, 1.0, 0.0, 1.0;
	0.0, 1.0, 0.0, 1.0;

	0.0, 1.0, 0.0, 1.0/3.0;
	0.0, 1.0, 1.0/3.0, 2.0/3.0;
	0.0, 1.0, 2.0/3.0, 1.0;
	0.0, 1.0, 0.0, 1.0/3.0;
	0.0, 1.0, 1.0/3.0, 2.0/3.0;
	0.0, 1.0, 2.0/3.0, 1.0;

	0.0, 1.0/2.0, 0.0, 1.0/2.0;
	0.0, 1.0/2.0, 1.0/2.0, 1.0;
	1.0/2.0, 1.0, 0.0, 1.0/2.0;
	1.0/2.0, 1.0, 1.0/2.0, 1.0;
	0.0, 1.0/2.0, 0.0, 1.0/2.0;
	0.0, 1.0/2.0, 1.0/2.0, 1.0;
	1.0/2.0, 1.0, 0.0, 1.0/2.0;
	1.0/2.0, 1.0, 1.0/2.0, 1.0;
    
];%%NO T DIMENSION

% global_config.extract_mid_level_descriptors.params.pooling_cells = [
%     0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
% 
% 	0.0, 1.0, 0.0, 1.0/3.0, 0.0, 1.0;
% 	0.0, 1.0, 1.0/3.0, 2.0/3.0, 0.0, 1.0;
% 	0.0, 1.0, 2.0/3.0, 1.0, 0.0, 1.0;
% 
% 	0.0, 1.0/2.0, 0.0, 1.0/2.0, 0.0, 1.0;
% 	0.0, 1.0/2.0, 1.0/2.0, 1.0, 0.0, 1.0;
% 	1.0/2.0, 1.0, 0.0, 1.0/2.0, 0.0, 1.0;
% 	1.0/2.0, 1.0, 1.0/2.0, 1.0, 0.0, 1.0;
% 
% 	0.0, 1.0, 0.0, 1.0, 0.0, 1.0/2.0;
% 	0.0, 1.0, 0.0, 1.0, 1.0/2.0, 1.0;
% 
% 	0.0, 1.0, 0.0, 1.0/3.0, 0.0, 1.0/2.0;
% 	0.0, 1.0, 1.0/3.0, 2.0/3.0, 0.0, 1.0/2.0;
% 	0.0, 1.0, 2.0/3.0, 1.0, 0.0, 1.0/2.0;
% 	0.0, 1.0, 0.0, 1.0/3.0, 1.0/2.0, 1.0;
% 	0.0, 1.0, 1.0/3.0, 2.0/3.0, 1.0/2.0, 1.0;
% 	0.0, 1.0, 2.0/3.0, 1.0, 1.0/2.0, 1.0;
% 
% 	0.0, 1.0/2.0, 0.0, 1.0/2.0, 0.0, 1.0/2.0;
% 	0.0, 1.0/2.0, 1.0/2.0, 1.0, 0.0, 1.0/2.0;
% 	1.0/2.0, 1.0, 0.0, 1.0/2.0, 0.0, 1.0/2.0;
% 	1.0/2.0, 1.0, 1.0/2.0, 1.0, 0.0, 1.0/2.0;
% 	0.0, 1.0/2.0, 0.0, 1.0/2.0, 1.0/2.0, 1.0;
% 	0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0/2.0, 1.0;
% 	1.0/2.0, 1.0, 0.0, 1.0/2.0, 1.0/2.0, 1.0;
% 	1.0/2.0, 1.0, 1.0/2.0, 1.0, 1.0/2.0, 1.0;
%     
% ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classification
global_config.classification.do = 1;
global_config.classification.func = 'm_classification_v3'; % v3 libsvm, v4:liblinear
global_config.classification.version = ...
    [global_config.extract_low_level_descriptors.version,'_low'];


global_config.classification.descriptor_type = {'low_level','mid_level'}; % {low_level} or {mid_level} or {low_level,mid_level} 
global_config.classification.descriptor_weight = [0,1];

% sparse LDA part
global_config.classification.do_sparseLDA = 0;

global_config.classification.sparse_level = 0.1;
global_config.classification.using_sparseLDA = 0;

global_config.classification.splits = [1];
global_config.classification.kernelmat.recompute = 1;
global_config.classification.libsvm_params.C = 4;
global_config.classification.liblinear_params.C = 1;
global_config.classification.rho = [0.5,0.5]; %[0.5,0.5,0.5,0.5,0.5,  0.5,0.5,0.5,0.5,0.5,  0.1,0.1,0.1,0.1,0.1];


global_config.classification.feature_types_mid = ...
    {'vc-2','vc-3'
%     'hof-1','hog-1','shape-1','mbhx-1','mbhy-1',...
%      'hof-2','hog-2','shape-2','mbhx-2','mbhy-2',...
%     'hof-3','hog-3','shape-3','mbhx-3','mbhy-3'
    };
%'hof-1','hog-1','shape-1','mbhx-1','mbhy-1', 'hof-2','hog-2','shape-2','mbhx-2','mbhy-2','hof-3','hog-3','shape-3','mbhx-3','mbhy-3'
global_config.classification.feature_types_low = {'vc'};%'shape','hog','hof','mbhx','mbhy'};
global_config.classification.class_idx = [1:15];  % spesify which class and videos to classification

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% global setting end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get the path of every part
cd ..;
global_config.root_path = pwd;
cd experiments;


global_config.dataset_path = fullfile(global_config.root_path,'datasets',global_config.dataset);
global_config.local_path = fullfile(global_config.root_path,'local',global_config.dataset);
global_config.splits_file_path = fullfile(global_config.root_path,'datasets','HMDB51_TestTrainMulti_7030_splits'); % case HMDB51


global_config.read_dataset_info.file_name = fullfile(global_config.local_path,sprintf('dataset_info_v%s.mat',global_config.read_dataset_info.version));
global_config.extract_features.path = fullfile(global_config.local_path,sprintf('features_v%s',global_config.extract_features.version));
global_config.learn_codebooks.path = fullfile(global_config.local_path,sprintf('codebooks_v%s',global_config.learn_codebooks.version));
global_config.coding.path = fullfile(global_config.local_path,sprintf('codes_v%s',global_config.coding.version));
global_config.extract_low_level_descriptors.path = fullfile(global_config.local_path,sprintf('low_level_descriptors_v%s',global_config.extract_low_level_descriptors.version));
global_config.clustering.path = fullfile(global_config.local_path,sprintf('clusters_v%s',global_config.clustering.version));
global_config.vc_clustering.path = fullfile(global_config.local_path,sprintf('clusters_v%s',global_config.vc_clustering.version));

global_config.extract_part_features.path = fullfile(global_config.local_path,sprintf('parts_features_v%s',global_config.extract_part_features.version));
global_config.learn_action_parts.path = fullfile(global_config.local_path,sprintf('parts_model_v%s',global_config.learn_action_parts.version));
global_config.extract_mid_level_descriptors.path = fullfile(global_config.local_path,sprintf('mid_level_descriptors_v%s',global_config.extract_mid_level_descriptors.version));
global_config.classification.path = fullfile(global_config.local_path,sprintf('classification_v_%s',global_config.classification.version));


fprintf('*******************************************************\n');
global_config = check_version(global_config);

if ~global_config.valid
    fprintf('invalid global_config, exist!\n');
    return;
end

fprintf('*******************************************************\n');
if (global_config.num_core > 1)
    if matlabpool('size') <= 0
        disp('Using paralle... Initialing ...');
        matlabpool('open','local',global_config.num_core);
        
    else
        disp('Using paralle... Already initialized...');
    end
else
    disp('NOT Using paralle... ');
    if matlabpool('size') > 0
        matlabpool close;
    end
end

%%
fprintf('*******************************************************\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1 %global_config.read_dataset_info.do % read data set info
    fprintf('doing: read_dataset_info\n');
    if ~(feval(global_config.read_dataset_info.func,global_config))
        fprintf('m_read_dataset_info wrong!\n');
        return;
    end
else
    fprintf('jumped: read_dataset_info\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
fprintf('*******************************************************\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 0,% global_config.extract_features.do
    fprintf('doing: extract_features\n');
    if ~(feval(global_config.extract_features.func,global_config))
        fprintf('extract_features wrong!\n');
        return;
    end
else
    fprintf('jumped: extract_features\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
fprintf('*******************************************************\n');
%%%%%%%%%%%%%%%%%%%%%%% low level  %%%%%%%%%%%%%%%%%%%%%%%%%
if 0, %global_config.learn_codebooks.do
    
    fprintf('doing: learn_codebooks\n');
    if ~(feval(global_config.learn_codebooks.func,global_config))
        fprintf('learn_codebooks wrong!\n');
        return;
    end
else
    fprintf('jumped: learn_codebooks\n');
end

%%
fprintf('*******************************************************\n');
if 0, %global_config.coding.do
    
    fprintf('doing: coding \n');
    if ~(feval(global_config.coding.func,global_config))
        fprintf('coding wrong!\n');
        return;
    end
else
    fprintf('jumped: coding\n');
    
end
%%
fprintf('*******************************************************\n');
if 0, %global_config.extract_low_level_descriptors.do
   
    fprintf('doing: extract_low_level_descriptors\n');
    if ~(feval(global_config.extract_low_level_descriptors.func,global_config))
        fprintf('extract_low_level_descriptors wrong!\n');
        return;
    end
else
    fprintf('jumped: extract_low_level_descriptors\n');
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%Load Visual Concepts%%%%%%%%%%%%%%%%%%%
fprintf('*******************************************************\n');
if 0,
   
    fprintf('doing: load low level features\n');
    if ~(feval(global_config.load_visual_concepts.func,global_config))
        fprintf('load visual concepts wrong!\n');
        return;
    end
else
    fprintf('jumped: load_visual_concepts\n');
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
fprintf('*******************************************************\n');
%%%%%%%%%%%%%%%%%%%%%%%% visual concept clustering %%%%%%%%%%%%%%%%%%%%%%%%%
if  0, %global_config.vc_clustering.do
   
    fprintf('doing: vc clustering\n');
    if ~(feval(global_config.vc_clustering.func,global_config))
        fprintf('vc clustering wrong!\n');
        return;
    end
else
    fprintf('jumped:vc clustering\n');
    
end

%%
fprintf('*******************************************************\n');
%%%%%%%%%%%%%%%%%%%%%%%% mid level %%%%%%%%%%%%%%%%%%%%%%%%%

%%
fprintf('*******************************************************\n');
if 0, %global_config.extract_part_features.do

    fprintf('doing: extract_part_features\n');
    if ~(feval(global_config.extract_part_features.func,global_config))
        fprintf('extract_part_features wrong!\n');
        return;
    end
else
    fprintf('jumped: extract_part_features\n');
end

%%
fprintf('*******************************************************\n');
if 0,%global_config.learn_action_parts.do
    
    fprintf('doing: learn_action_parts\n');
    if ~(feval(global_config.learn_action_parts.func,global_config))
        fprintf('learn_action_parts wrong!\n');
        return;
    end
else
    fprintf('jumped: learn_action_parts\n');
end
%%%%%%%%%%%%%%%%%%%%
%Updated here%%%%%%%
%%%%%%%%%%%%%%%%%%%%

%%
fprintf('*******************************************************\n');
if 0,% global_config.extract_mid_level_descriptors.do
    
    fprintf('doing: extract_mid_level_descriptors\n');
    if ~(feval(global_config.extract_mid_level_descriptors.func,global_config))
        fprintf('extract_mid_level_descriptors wrong!\n');
        return;
    end
else
    fprintf('jumped: extract_mid_level_descriptors\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
fprintf('*******************************************************\n');
%%%%%%%%%%%%%%%%%%%%%%% classification %%%%%%%%%%%%%%%%%%%%%%
if global_config.classification.do
    
    fprintf('doing: classification descriptor:low_level:%s mid_level %s\n',global_config.extract_low_level_descriptors.version,global_config.extract_mid_level_descriptors.version);
    if ~(feval(global_config.classification.func,global_config))
        fprintf('classification wrong!\n');
        return;
    end
else
    fprintf('jumped: classification\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%