import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import kl_divergence
from torch.autograd import Variable


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, vae_hidden_dim, z_dim):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, vae_hidden_dim)
        self.fc2 = nn.Linear(vae_hidden_dim, z_dim)  # mu
        self.fc3 = nn.Linear(vae_hidden_dim, z_dim)  # log_var
        self.relu = nn.ReLU()

    def reparameterization(self, mu, log_var, test_mode=False):
        if(test_mode):
            return mu
        else:
            std = th.exp(0.5 * log_var)
            eps = th.randn_like(std)
            return mu + std * eps

    def forward(self, input, test_mode=False):
        h = self.relu(self.fc1(input))
        mu = self.fc2(h)
        log_var = th.clamp(self.fc3(h), min=-20, max=20)
        sampled_z = self.reparameterization(mu, log_var, test_mode=test_mode)
        return sampled_z, mu, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, z_dim, vae_hidden_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, vae_hidden_dim)
        self.fc2 = nn.Linear(vae_hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_z):
        h = self.relu(self.fc1(input_z))
        return self.fc2(h)


class Compute_Q(nn.Module):
    def __init__(self, rnn_hidden_dim, z_dim, output_dim, n_agents, args):
        super(Compute_Q, self).__init__()
        self.fc1 = nn.Linear(rnn_hidden_dim + z_dim + n_agents, 16)
        self.fc2 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.n_agents = n_agents
        self.args = args

    # z: [bs, v] or [bs, n ,v], h:[bs * n, rnn_hidden_dim]
    def forward(self, h, z):
        q = self.fc1_layer(h, z)
        q = self.fc2(self.relu(q))
        return q

    def fc1_layer(self, h, z):
        bs = z.shape[0]
        if z.dim() == 2:  # test_mode=False
            z = th.repeat_interleave(z, self.n_agents, dim=0)
        else:  # test_mode=True
            z = z.reshape(bs * self.n_agents, -1)
        agent_id = th.eye(self.n_agents).cuda().unsqueeze(
            0).expand(bs, -1, -1).reshape(bs * self.n_agents, -1)
        return self.fc1(th.cat((z, h, agent_id), dim=1))


class POE(nn.Module):
    def __init__(self, input_shape, vae_hidden_dim, z_dim, n_agents, args):
        super(POE, self).__init__()
        self.message_encoder = VAE_Encoder(input_shape, vae_hidden_dim, z_dim)
        self.n_agents = n_agents
        self.z_dim = z_dim
        self.args = args

    def reparameterization(self, mu, log_var, test_mode=False):
        if(test_mode):
            return mu
        else:
            std = th.exp(0.5 * log_var)
            eps = th.randn_like(std)
            return mu + std * eps

    # inputs: [bs, n, hdim]
    def infer(self, inputs, eps=1e-7):
        bs = inputs.shape[0]
        p_mu, p_logvar = self.prior_expert((bs, 1, self.z_dim))
        mus = []
        logvars = []
        for i in range(self.n_agents):
            input = inputs[:, i]  # [bs, h_dim]
            _, mu, logvar = self.message_encoder(
                input)  # [bs, z_dim]
            mus.append(mu)
            logvars.append(logvar)
        mus = th.stack(mus, dim=1)  # [bs, n, z_dim]
        logvars = th.stack(logvars, dim=1)
        mu = th.cat((p_mu, mus), dim=1)
        logvar = th.cat((p_logvar, logvars), dim=1)
        # POE
        var = th.exp(logvar) + eps  # logvar [bs , n, z_dim]
        T = 1. / (var + eps)
        poe_mu = th.sum(mu * T, dim=1) / th.sum(T, dim=1)
        poe_var = 1. / th.sum(T, dim=1)
        poe_logvar = th.log(poe_var + eps)  # [bs, z_dim]
        return poe_mu, poe_logvar

    #input: [n, h_dim]
    def apply_each_agent(self, inputs, eps=1e-7):
        p_mu, p_logvar = self.prior_expert((1, self.z_dim))
        mus = []
        logvars = []
        for i in range(self.n_agents):
            input = inputs[i]  # [h_dim]
            _, mu, logvar = self.message_encoder(
                input)  # [z_dim]
            mus.append(mu)
            logvars.append(logvar)
        mus = th.stack(mus)  # [n, z_dim]
        logvars = th.stack(logvars)
        mu = th.cat((p_mu, mus), dim=0)
        logvar = th.cat((p_logvar, logvars), dim=0)
        # POE
        var = th.exp(logvar) + eps  # [n, z_dim]
        T = 1. / (var + eps)
        poe_mu = th.sum(mu * T, dim=0) / th.sum(T, dim=0)
        return poe_mu

    def forward(self, hidden_inputs, test_mode=False):
        poe_mu, poe_logvar = self.infer(hidden_inputs)
        z = self.reparameterization(
            poe_mu, poe_logvar, test_mode=test_mode)  # z: [bs, z_dim]
        return z, poe_mu, poe_logvar

    def prior_expert(self, size):
        mu = Variable(th.zeros(size)).cuda()
        logvar = Variable(th.zeros(size)).cuda()
        return mu, logvar


class CroMAC(nn.Module):
    def __init__(self, input_shape, args, scheme):
        super(CroMAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.state_dim = scheme["state"]["vshape"]
        self.obs_dim = scheme["obs"]["vshape"]
        self.state_encoder = VAE_Encoder(
            self.state_dim, args.vae_hidden_dim, args.z_dim)
        self.state_decoder = VAE_Decoder(
            args.z_dim, args.vae_hidden_dim, self.state_dim)
        self.mse_loss = nn.MSELoss(reduce=False)
        self.poe = POE(self.args.rnn_hidden_dim,
                       args.vae_hidden_dim, args.z_dim, self.n_agents, args)
        self.compute_q = Compute_Q(args.rnn_hidden_dim,
                                   args.z_dim, args.n_actions, args.n_agents, args)
        self.relu = nn.ReLU()
        self.ce = nn.CrossEntropyLoss()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, state, hidden_state, test_mode=False, train_mode=False, actions=None, mask=None, t_env=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        if test_mode:
            if self.args.noise:
                q = []
                for agent_id in range(self.n_agents):
                    h_copy = h.clone().detach()
                    h_copy.requires_grad = True
                    ori_zs, _, _ = self.poe(
                        h_copy.reshape(-1, self.n_agents, self.args.rnn_hidden_dim), test_mode=test_mode)  # [1, z_dim]
                    ori_q = self.compute_q(h_copy, ori_zs)
                    ori_label = th.argmax(ori_q, dim=1)
                    h_adv_i = self.FGSMAdv(self.compute_q,
                                           agent_id, h_copy, ori_label, ori_zs)
                    q_i = self.compute_each_q(h_adv_i, agent_id)
                    q.append(q_i)
                q = th.stack(q)
            else:
                z_poe, _, _ = self.poe(
                    h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim), test_mode=test_mode)
                q = self.compute_q(h, z_poe)
        else:
            z_state, z_s_mu, z_s_logvar = self.state_encoder(
                state, test_mode=test_mode)
            q = self.compute_q(h, z_state)
        losses = {}
        if train_mode:
            recon_state = self.state_decoder(z_state)
            losses["vae_state_loss"] = self.VAE_loss_func(
                recon_state, state, z_s_mu, z_s_logvar)
            losses["poe_loss"] = self.PoE_loss_func(
                h, z_s_mu.detach(), z_s_logvar.detach(), test_mode=test_mode)
            if self.args.robust and t_env > self.args.robust_start_time:
                losses["robust_loss"] = self.robust_loss_func(
                    h, z_s_mu.detach(), actions, mask)
        return q, h, losses

    def FGSMAdv(self, model, i, ori_input, ori_label, z):
        # ori_input: [n, h_dim]
        assert(ori_input.dim() == 2)
        h = ori_input[i]  # [1, h_dim]
        agent_id = th.zeros(self.n_agents).cuda()
        agent_id[i] = 1
        q = model.fc1(th.cat((z.reshape(-1), h, agent_id)))
        q = model.fc2(model.relu(q)).reshape(1, -1)
        loss = self.ce(q, ori_label[i].reshape(-1))
        self.zero_grad()
        loss.backward()
        return ori_input + self.args.noise_epsilon*th.sign(ori_input.grad)

    def compute_each_q(self, input, i):
        h = input[i]
        z = self.poe.apply_each_agent(input)
        agent_id = th.zeros(input.shape[0]).cuda()
        agent_id[i] = 1
        q = self.compute_q.fc1(th.cat((z, h, agent_id)))
        q = self.compute_q.fc2(self.relu(q))
        return q

    def VAE_loss_func(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x).sum(-1).mean()
        KLD = -0.5 * th.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1).mean()
        return (MSE + KLD) * self.args.vae_loss_weight

    def PoE_loss_func(self, h, z_s_mu, z_s_logvar, test_mode):
        _, poe_mu, poe_logvar = self.poe(h.detach(
        ).reshape(-1, self.n_agents, self.args.rnn_hidden_dim), test_mode=test_mode)
        p = D.Normal(z_s_mu, th.exp(0.5 * z_s_logvar))
        q = D.Normal(poe_mu, th.exp(0.5 * poe_logvar))
        KLD = kl_divergence(p, q).sum(-1).mean()
        if th.isnan(KLD):
            th.set_printoptions(threshold=3000)
            print(z_s_mu)
            print(z_s_logvar)
            print(poe_mu)
            print(poe_logvar)
            assert(0)
        return KLD * self.args.poe_loss_weight

    def robust_loss_func(self, h, z_s_mu, actions, mask):
        bs = z_s_mu.shape[0]
        q_values = self.compute_q(h, z_s_mu).reshape(bs, self.n_agents, -1)
        mu = self.compute_q.fc1_layer(h, z_s_mu)
        r = th.ones(bs * self.n_agents, self.args.z_dim +
                    self.args.rnn_hidden_dim).cuda() * self.args.noise_epsilon * self.args.kappa
        agent_id = th.eye(self.n_agents).cuda().unsqueeze(
            0).expand(bs, -1, -1).reshape(bs * self.n_agents, -1)
        r_add_id = th.cat((r, agent_id), dim=-1)
        r = F.linear(r_add_id, th.abs(self.compute_q.fc1.weight))
        upper = self.relu(mu + r)
        lower = self.relu(mu - r)
        prev_mu = (upper + lower)/2
        prev_r = (upper - lower)/2
        mu = self.compute_q.fc2(prev_mu)
        r = F.linear(prev_r, th.abs(self.compute_q.fc2.weight))
        upper = (mu + r).reshape(bs, self.n_agents, -1)
        lower = (mu - r).reshape(bs, self.n_agents, -1)
        q_diff = th.max(th.zeros([1]).cuda(),
                        (q_values.gather(2, actions).detach()-q_values.detach()))
        overlap = th.max(upper - lower.gather(2, actions
                                              ), th.zeros([1]).cuda())
        worst_case_loss = th.sum(
            q_diff*overlap*mask.expand(bs, self.n_agents).reshape(bs, self.n_agents, -1).expand(bs, self.n_agents, self.args.n_actions), dim=2).mean()
        return worst_case_loss * self.args.robust_weight
