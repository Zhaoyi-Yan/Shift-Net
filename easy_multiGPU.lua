-- It is an easy script for multi-gpu training in Torch7.

require 'optim'
require 'nn'
require 'cutorch'
require 'cunn'
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

local conv_net = nn.Sequential()
local concat = nn.ConcatTable()
concat:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
concat:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(concat)
local jt = nn.JoinTable(2)
conv_net:add(jt)
conv_net:add(nn.SpatialConvolution(6,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1):noBias())
conv_net:cuda()

local optimState = {
    learningRate = 1,
    momentum = 0,
}

num_epochs = 1
batchsize = 64
criterion = nn.MSECriterion():cuda()
input = torch.randn(1000,3,256,256):mul(100):floor():cuda()
target = torch.randn(1000,3,256,256):mul(100):floor():cuda()
num_takes = input:size(1)/batchsize


gpus = torch.range(1, 2):totable() 
dpt = nn.DataParallelTable(1, true)
    :add(model,gpus)
    :threads(function()
    require 'nngraph'
        local cudnn = require 'cudnn'
        cudnn.fastest, cudnn.benchmark = true, true
        end)
dpt.gradInput = nil  -- save memory
model = dpt:cuda()
dpt:add(conv_net, {1,2})
print(dpt)

-- TRAINING:
local timer = torch.Timer()
for i = 1, num_epochs do
  for bth = 1, num_takes do
    local input_bt = input:narrow(1, (bth-1)*batchsize+1, batchsize)
    local target_bt = target:narrow(1, (bth-1)*batchsize+1, batchsize)
    print('Processing epoch: ',i,' , batch: ',bth)
   -- net:syncParameters() -- It can be omitted.

    local function feval(net)
        local params, gradParams = net:getParameters()
        return params, function(x)
            net:zeroGradParameters()
            local output = net:forward(input_bt)
            local err = criterion:forward(output, target_bt)
            local gradOutput = criterion:backward(output, target_bt)
            local gradInput = net:backward(input_bt, gradOutput)
            return err, gradParams
        end
    end

    local params_dpt, feval_dpt = feval(dpt)

    optim.sgd(feval_dpt, params_dpt, optimState)
    -- net:syncParameters() -- It can be omitted.

  end
end
print('Totally elapse time: '..timer:time().real..' seconds')

-- For more information, please refer to https://github.com/soumith/imagenet-multiGPU.torch
