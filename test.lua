require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

require 'lib/MaxCoord'
require 'lib/InstanceNormalization'
require 'lib/innerTripleShift'
require 'lib/myUtils'
require 'lib/innerCos'

opt = {
    mask_type = 'center',     -- random or center. Should consistent with training setting.
    fixed_mask = true,        -- whether the mask is fixed.
    DATA_ROOT = './datasets/Paris_StreetView_Dataset/', -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 1,            -- # images in batch
    loadSize = 256,           -- scale images to this size
    fineSize = 256,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    display = 0,              -- display samples while training. 0 = false
    display_id = 200,         -- display window id.
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
    which_direction = 'AtoB', -- AtoB or BtoA
    phase = 'paris_test',     -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = 'paris_train_shiftNet',  -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints/', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = '30',            -- which epoch to test? set to 'latest' to use latest cached model
    overlap = 4,
}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'

local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- translation direction
local idx_A = nil
local idx_B = nil
local input_nc = opt.input_nc
local output_nc = opt.output_nc

idx_A = {1, input_nc}
idx_B = {input_nc+1, input_nc+output_nc}

----------------------------------------------------------------------------
local mask_global = torch.ByteTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local target = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)


printNet(netG)

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

if opt.how_many=='all' then
    opt.how_many=math.huge
end
opt.how_many=math.min(opt.how_many, data:size())

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



local filepaths = {} -- paths to images tested on
-- save fixed mask
if opt.fixed_mask == true and opt.mask_type == 'random' then
    mask_global = load_mask('v_mask.png', opt.fineSize)
end
for n=1,math.floor(opt.how_many/opt.batchSize) do
    print('processing batch ' .. n)
    
    local data_curr, filepaths_curr = data:getBatch()
    filepaths_curr = util.basename_batch(filepaths_curr)
    print('filepaths_curr: ', filepaths_curr)
    
    mask_global = mask_global:byte()
    if opt.mask_type == 'random' and opt.fixed_mask == false then
        mask_global = torch.ByteTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        mask_global = create_gMask(pattern, mask_global, MAX_SIZE, opt)
    elseif opt.mask_type == 'center' then
        mask_global = mask_global:zero()
        mask_global[{{},{},{1 + opt.fineSize/4+opt.overlap, opt.fineSize/2 + opt.fineSize/4-opt.overlap},{1 + opt.fineSize/4+opt.overlap, opt.fineSize/2 + opt.fineSize/4-opt.overlap}}] = 1
    end

    input = data_curr[{ {}, idx_A, {}, {} }]:clone()
    target = data_curr[{ {}, idx_A, {}, {} }]:clone()

    input[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
    input[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
    input[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0



    if opt.gpu > 0 then
        input = input:cuda()
        mask_global = mask_global:cuda()
    end

    local timer = torch.Timer()
    local output_tb_pre = netG:forward({input, mask_global})
    print(timer:time().real ..' seconds')
    output = util.deprocess_batch(output_tb_pre[1])
    input = util.deprocess_batch(input):float()

    --  Make the center of input `blank`
    input[{{},{1},{},{}}][mask_global:float():byte()] = 1.0
    input[{{},{2},{},{}}][mask_global:float():byte()] = 1.0
    input[{{},{3},{},{}}][mask_global:float():byte()] = 1.0


    output = output:float()
    target = util.deprocess_batch(target):float()

    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    paths.mkdir(paths.concat(image_dir,'input'))
    paths.mkdir(paths.concat(image_dir,'output'))
    paths.mkdir(paths.concat(image_dir,'target'))

    for i=1, opt.batchSize do
        image.save(paths.concat(image_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'target',filepaths_curr[i]), image.scale(target[i],target[i]:size(2),target[i]:size(3)/opt.aspect_ratio))
    end
    print('Saved images to: ', image_dir)
    
    if opt.display then
      if opt.preprocess == 'regular' then
        disp = require 'display'
        disp.image(util.scaleBatch(input,100,100),{win=opt.display_id, title='input'})
        disp.image(util.scaleBatch(output,100,100),{win=opt.display_id+1, title='output'})
        disp.image(util.scaleBatch(target,100,100),{win=opt.display_id+2, title='target'})
        
        print('Displayed images')
      end
    end
    filepaths = TableConcat(filepaths, filepaths_curr)
end

-- make webpage
io.output(paths.concat(opt.results_dir,opt.netG_name .. '_' .. opt.phase, 'index.html'))

io.write('<table style="text-align:center;">')

io.write('<tr><td>Image #</td><td>Input</td><td>Output</td><td>Ground Truth</td></tr>')
for i=1, #filepaths do
    io.write('<tr>')
    io.write('<td>' .. filepaths[i] .. '</td>')
    io.write('<td><img src="./images/input/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/output/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/target/' .. filepaths[i] .. '"/></td>')
    io.write('</tr>')
end

io.write('</table>')