require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'cudnn'
require 'cunn'
require 'lib/MaxCoord'
require 'lib/InstanceNormalization'
require 'lib/innerTripleShift'
require 'lib/myUtils'
require 'lib/innerCos'

opt = {
   mask_type = 'center',            -- 'center' or 'random'
   fixed_mask = true,               -- whether the mask is fixed.
   name = 'paris_train_shiftNet',              -- name of the experiment, should generally be passed on the command line
   DATA_ROOT = './datasets/Paris_StreetView_Dataset/',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- images in batch
   loadSize = 350,         -- scale images to this size
   fineSize = 256,         -- then crop to this size
   ngf = 64,               -- of gen filters in first conv layer
   ndf = 64,               -- of discrim filters in first conv layer
   input_nc = 3,           -- of input image channels
   output_nc = 3,          -- of output image channels
   niter = 30,             -- of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_plot = 'errL1, errG, errD',  -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   phase = 'paris_train',       -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 2,         -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 10000,   -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   precise_model_num = true,
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 0,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'shiftNet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective
   threshold = 5/16,             -- making binary mask
   stride = 1,         -- should be dense, 1 is a good option.
   shift_size = 1,     -- when it is 3, the next opt `mask_thred` should not be 1
   mask_thred = 1,     -- define when how many `1`s can make the patch masked.
   bottleneck = 512,   -- bottleneck of netG
   constrain = 'MSE',  --  the type of guidance loss
   strength = 1,       -- the weight of guidance loss
   triple_weight = 1,  -- the weight of the third gradient on the first part gradient.
   overlap = 4,        -- set 4 just to keep consistent with context-encoders.
   gan_weight = 0.2,   -- additional weight of gan loss
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

require 'cutorch'
cutorch.setDevice(opt.gpu)


local input_nc = opt.input_nc
local output_nc = opt.output_nc


local idx_A = nil
local idx_B = nil

idx_A = {1, input_nc}
idx_B = {input_nc+1, input_nc+output_nc}

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

-- init artificial swapX to avoid error
local res = 0.06 -- the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
local density = 0.25
local MAX_SIZE = 10000
local low_pattern = torch.Tensor(res*MAX_SIZE, res*MAX_SIZE):uniform(0,1):mul(255)
local pattern = image.scale(low_pattern, MAX_SIZE, MAX_SIZE,'bicubic')
low_pattern = nil
collectgarbage()
pattern:div(255)
pattern = torch.lt(pattern,density):byte()  -- 25% 1s and 75% 0s
pattern = pattern:byte()
print('...Random pattern generated')

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local constrainCriterion    -- add constrain criterion
if opt.constrain == 'MSE' then constrainCriterion = nn.MSECriterion() else constrainCriterion = nn.AbsCriterion() end
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0


function defineG(input_nc, output_nc, ngf)

    local shiftTripleLayer = nn.InnerTripleShift('innerTripleShift_32',opt.shift_size, opt.threshold, opt.stride, opt.mask_thred, opt.triple_weight, opt.fixed_mask)
    local GT_latent_fake = torch.CudaTensor(opt.batchSize, ngf*4, 32, 32)
    local addCos32 = nn.AddCos('addCos_32_fake', constrainCriterion, opt.threshold, GT_latent_fake, opt.strength)


    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "shiftNet" then netG = defineG_shiftNet(input_nc, output_nc, ngf,  shiftTripleLayer, addCos32, opt.bottleneck)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)
    --reassgin the correct maskModel weight
    printNet(netG)
    netG.modules[41].weight:fill(1/16)
    netG.modules[42].weight:fill(1/16)
    netG.modules[43].weight:fill(1/16)
    return netG
end



function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels 
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end

-- load saved models and finetune
local start_epoch = 0
if opt.continue_train == 1 then
    local current_dir = paths.concat(opt.checkpoints_dir, opt.name)

    local continue_txt = 'continue.txt'
    if io.open(continue_txt,'r') ~= nil then
        os.execute('rm -r '..continue_txt)
    end

    os.execute('cd '..current_dir..';'..'ls -d *.t7 | tee '..continue_txt)
    local file_continue = io.open(current_dir..'/'..continue_txt,'r')
    local file_content = io.open(current_dir..'/'..continue_txt,'r')
    local latest_saved_num = 0
    local file_content_all = file_content:read('*a')
    if file_content_all ~= '' then
        for line in file_continue:lines() do
            local st, _ = string.find(line, '%d_net_G.t7')
            if st then  -- avoid latest.t7
                local tmp = tonumber(string.sub(line,1, st))
                if tmp > latest_saved_num then
                    latest_saved_num = tmp
                end
            end
        end
        local load_model_prefix = nil
        if latest_saved_num == 0 then
            load_model_prefix = 'latest'
            print('Warning: it seems that no models whose names contains numbers pretrained, so just train with start index 1')
        end
        if load_model_prefix == nil then
            load_model_prefix = tostring(latest_saved_num)
        end
        print('Epoch starting at '..latest_saved_num+1)
        start_epoch = latest_saved_num

        local exist_latest = io.open(current_dir..'/'..'latest_net_G.t7')
        if opt.precise_model_num == true then
            if latest_saved_num == 0 then
                error('no models whose names contains numbers pretrained')
            end
        else
            if exist_latest == nil then
                error('No \'latest\' models saved!')
            else
                load_model_prefix = 'latest'
                exist_latest:close()
            end
        end
        print('loading previously trained netG...')
        netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, load_model_prefix..'_net_G.t7'), opt)
        print('loading previously trained netD...')
        netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, load_model_prefix..'_net_D.t7'), opt) 
    else
        error('no pretrained model, you\'d better train from scratch')
    end
    file_continue:close()
    file_content:close()
else
    print('define model netG...')
    netG = defineG(input_nc, output_nc, ngf)
    print('define model netD...')
    netD = defineD(input_nc, output_nc, ndf)
end

print('netD...')
print(netD)


local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local GT_latent_32 = nil
local mask_global = torch.ByteTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B1 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   if opt.cudnn==1 and opt.continue_train == 0 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
   print('done')
else
	print('running model on CPU')
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end



function createRealFake()
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    real_data = real_data:cuda()
    data_tm:stop()

    real_A:copy(real_data[{ {}, idx_A, {}, {} }])

    if opt.mask_type == 'random' then
        mask_global = create_gMask(pattern, mask_global, MAX_SIZE, opt)
    elseif opt.mask_type == 'center' then
        mask_global = mask_global:zero()
        mask_global[{{},{},{1 + opt.fineSize/4 + opt.overlap, opt.fineSize/2 + opt.fineSize/4 - opt.overlap},
        {1 + opt.fineSize/4 + opt.overlap, opt.fineSize/2 + opt.fineSize/4 - opt.overlap}}] = 1
    end

    real_A[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
    real_A[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
    real_A[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0  

    real_B:copy(real_data[{ {}, idx_A, {}, {} }])

   
    -- real_A is input(we use GAN preliminary filled images), and real_B is groundTruth
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end

    -- transfer mask to cudaTensor()
    mask_global = mask_global:cuda()

    netG:forward({real_B, mask_global})
    GT_latent_32 = netG.modules[46].latent_in_mask -- get the encoded feature of GT.

    -- constructe the real addCos and replace the fake one with the real one.
    local addCos32_real = nn.AddCos('addCos_32_real', constrainCriterion, opt.threshold, GT_latent_32, opt.strength)
    netG:replace(function(module)
        if  module.cls == 'addCos_32_fake' then
            return addCos32_real:cuda()
        else
            return module
        end
    end)

    local fake_B_tb = netG:forward({real_A, mask_global})
    fake_B = fake_B_tb[1]

    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
    
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    -- train netD with (real, real_label)
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)
    
    -- Fake
    -- train netD with (fake_AB, fake_label)
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    
    -- output are netD:forward(fake_AB), just a serials of labels of float number.
    -- then We need to minimize the loss between output and real_label
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
       local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
       if opt.gpu>0 then 
       	label = label:cuda();
       end
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       -- If we use cGAN, then assume that the grad is bs*6*h*w, then we only need the grad 
       -- of fake_B. So narrow(2, ....)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
    else
        errG = 0
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_B, real_B)
       df_do_AE = criterionAE:backward(fake_B, real_B)
    else
        errL1 = 0
    end

    local df_dummy = torch.Tensor(GT_latent_32:size()):typeAs(GT_latent_32):zero()
    netG:backward({real_A, mask_global}, {df_dg:mul(opt.gan_weight) + df_do_AE:mul(opt.lambda), df_dummy})

    return errG, gradParametersG
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
    if not util.containsValue({"errG", "errD", "errL1"}, v) then 
        error(string.format('bad display_plot value "%s"', v)) 
    end
end

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

local counter = 0
for epoch = start_epoch+1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end


        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),200,200)), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),200,200)), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),200,200)), {win=opt.display_id+2, title=opt.name .. ' target'})
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG, errD, errL1))
           
            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
              if loss[v] ~= nil then
               plot_vals[#plot_vals + 1] = loss[v] 
             end
            end

            -- update display plot
            if opt.display then
              table.insert(plot_data,plot_vals )--{epoch, errG,errD,errG_l2}
              plot_config.win = plot_win
              plot_win = disp.plot(plot_data, plot_config)
            end
        
        -- save latest model
          if counter % opt.save_latest_freq == 0 then
              print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
          end
      end
    end
    
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    collectgarbage()
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()

end
