local AddCos, parent = torch.class('nn.AddCos', 'nn.Module')

function AddCos:__init(cls, criterion, threshold, GT_latent, strength)
    parent.__init(self)

    self.cls = cls
    self.criterion = criterion
    self.threshold = threshold
    self.GT_latent = GT_latent
    self.strength = strength
end

-- It abandoms the last channel(mask)
function AddCos:updateOutput(input)
    assert(input:nDimension() == 4)
    self.bz = input:size(1)
    self.c = input:size(2)
    self.h = input:size(3)
    self.w = input:size(4)

    self.real_c = self.c - 1
    local input_latent = input:narrow(2, 1, self.real_c)  -- It is `self.bz*self.real_c*self.h*self.w`
    local mask_float = input:narrow(2, self.c, 1):narrow(1,1,1):squeeze()  -- `self.h*self.w`
    local mask_bin = binary_mask(mask_float, self.threshold)

    self.inv_mask_bin = torch.add(mask_bin:float():neg(), 1):byte()

    self.inv_mask_bin = self.inv_mask_bin:repeatTensor(self.bz, self.real_c, 1, 1)
    
    self.GT_latent_mask = self.GT_latent:clone()
    local GT_latent_h = self.GT_latent_mask:size(3)
    local GT_latent_w = self.GT_latent_mask:size(4)

    if GT_latent_h ~= self.h or GT_latent_w ~= self.w then
        self.GT_latent_mask:resize(self.bz, self.real_c, self.h, self.w)
    end
    self.GT_latent_mask[self.inv_mask_bin] = 0

    self.input_latent_mask = input_latent:clone()
    self.input_latent_mask[self.inv_mask_bin] = 0

    self.loss = self.criterion:forward(self.input_latent_mask, self.GT_latent_mask)
    self.loss = self.loss * self.strength

    self.output = input:narrow(2, 1, self.real_c)  -- discard the last mask dimension

    return self.output
end

-- Just push dummy_mask_grad(filled with 0s) to the last channel.
function AddCos:updateGradInput(input, gradOutput)
    local gradInput_former = self.criterion:backward(self.input_latent_mask, self.GT_latent_mask)
    gradInput_former:mul(self.strength)
    gradInput_former:add(gradOutput)

    local dummy_mask_grad = torch.Tensor(self.bz, 1, self.h, self.w):typeAs(input):zero()
    self.gradInput = torch.cat({gradInput_former, dummy_mask_grad}, 2)
    return self.gradInput
end

function AddCos:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()
end

