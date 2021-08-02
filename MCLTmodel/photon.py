import numpy as np

THRESHOLD = 1e-4  # critical weight for roulette
CHANCE = 0.1  # Chance of roulette survival
COSZERO = 1.0 - 1.0e-12  # cosine of about 1e-6 rad
COS90D = 1.0e-6  # cosine of about 1.57 - 1e-6 rad
# COS90D = .000001

def RFresnel(n1, n2, ca1):
    # Compute the Fresnel reflectance.
    # Make sure that the cosine of the incident angle a1
    # is positive, and the case when the angle is greater
    # than the critical angle is ruled out.
    # Avoid trigonometric function operations as much as
    # possible, because they are computation-intensive.

    if n1 == n2:  # matched boundary
        ca2 = ca1
        r = 0.0
    elif ca1 > COSZERO:  # normal incident
        ca2 = ca1
        r = (n2 - n1) / (n2 + n1)
        r *= r
    elif ca1 < COS90D:  # very slant
        ca2 = 0.0
        r = 1.0
    else:  # general
        # sine of the incident and transmission angles
        sa1 = (1.0 - ca1 * ca1) ** 0.5
        sa2 = n1 * sa1 / n2
        if sa2 >= 1.0:
            # double check for total internal reflection
            ca2 = 0.0
            r = 1.0
        else:
            # cosines of the sum ap or
            # difference am of the two
            # angles. ap = a1+a2
            # am = a1 - a2
            ca2 = (1.0 - sa2 * sa2) ** 0.5
            cap = ca1 * ca2 - sa1 * sa2  # c+ = cc - ss
            cam = ca1 * ca2 + sa1 * sa2  # c- = cc + ss
            sap = sa1 * ca2 + ca1 * sa2  # s+ = sc + cs
            sam = sa1 * ca2 - ca1 * sa2  # s- = sc - cs
            r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam)
    return r, ca2


def SpinTheta(g):
    # Choose (sample) a new theta angle for photon propagation
    # according to the anisotropy.
    # If anisotropy g is 0, then
    # cos(theta) = 2*rand-1.
    # otherwise
    # sample according to the Henyey-Greenstein function.
    # Returns the cosine of the polar deflection angle theta.
    if g == 0.0:
        cost = 2 * np.random.random_sample() - 1
    else:
        temp = (1 - g * g) / (1 - g + 2 * g * np.random.random_sample())
        cost = (1 + g * g - temp * temp) / (2 * g)
        if cost < -1:
            cost = -1.0
        elif cost > 1:
            cost = 1.0
    return cost

#launching photons

class Photon:
    """Photon class - Collimated launch.
        Class variables:
            Launch parameters:
                x - Cartesian coordinate x [cm]
                y - Cartesian coordinate y [cm]
                z - Cartesian coordinate z [cm]
                ux - directional cosine x of a photon
                uy - directional cosine y of a photon
                uz - directional cosine z of a photon
            Tracking parameters:
                w - weight
                alive - False if photon is terminated
                s - current step size [cm]
            Output parameter:
                output - tuple (ir, w) with index of reflected photon location and weight
        Instance variables:
            Tissue parameters:
                mua - absorption coefficient [1/cm]
                mus - scattering coefficient [1/cm]
                g - anisotropy factor
                n - refractive index
            Spatial boundary parameters:
                d - thickness of slab [cm]
                nf - refractive index external medium frontside of sample
                nr - refractive index external medium rearside of sample
            Grid parameters:
                Nz - number of z bins (depth)
                Nr - number of r bins (depth)
                Na - number of alpha bins (depth)
                dr - r grid separation [cm]

        Methods:
            Hop-Drop-Spin-Roulette
            as described in paper by
    """

    def __init__(self, mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr):
        # initialize a photon
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        # self.trajectory = [(self.x, self.y, self.z)]
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 1.0
        self.w = 1.0
        self.alive = True
        self.output = (0, 0)

        self.mua = mua
        self.mus = mus
        self.mut = mua + mus
        self.g = g
        self.n = n

        self.d = d
        self.nf = nf
        self.nr = nr

        self.Nz = Nz
        self.Nr = Nr
        self.Na = Na
        self.dz = d / Nz #z grid separation [cm]
        self.dr = dr
        self.da = 2 * np.pi / Na #alpha grid separation [radian]

        self.Nscattered = 0


        # critical angle at top interface of the current layer
        if n > nf:
            self.cosCrit0 = (1.0 - nf * nf / (n * n)) ** 0.5
        else:
            self.cosCrit0 = 0.0
        # crticial angle at bottom interface of the current layer
        if n > nr:
            self.cosCrit1 = (1.0 - nr * nr / (n * n)) ** 0.5
        else:
            self.cosCrit1 = 0.0



    def Propagate(self):
        while self.alive:
            self.Hop()
            self.Drop()
            self.Spin()
            if (self.w < THRESHOLD) and self.alive:
                self.Roulette()

    def Hop(self):
        s = -np.log(np.random.random()) / self.mut

        self.x += s * self.ux
        self.y += s * self.uy
        self.z += s * self.uz

        # Check boundaries
        uz = self.uz

        if self.z < 0:  #if photon crosses front surface

            #retract step
            self.x -= s * self.ux
            self.y -= s * self.uy
            self.z -= s * self.uz

            #compute new direction and reflection probability
            if -uz <= self.cosCrit0:   #total internal reflection based on critical angle
                r = 1.0
            else:
                r, uz1 = RFresnel(self.n, self.nf, -uz)

            if np.random.random_sample() > r:  # moves out of medium through front surface -> reflection
                self.uz = -uz1

                ir = int((self.x*self.x + self.y*self.y)**.5/self.dr)
                # # check if out of bounds -> move to last 'overflow' bin

                if ir > (self.Nr - 1):
                    ir = self.Nr - 1
                self.output = (ir, self.w)
                # R_r[ir] += self.w  #this line can be unhashed if you want to run a script with global Reflection array variable

                self.status = 'Reflected'
                self.alive = False
            else:  # move back in the medium (internal reflection)
                self.x += s * self.ux
                self.y += s * self.uy
                self.z = - (self.z + s * self.uz) #mirror

                self.uz = -uz  #change in opposite z-direction (from + to -)

        elif self.z > self.d:  # if photon crosses rear surface (completely reflecting background)

            # retract step
            self.z -= s * self.uz
# move back in the medium (internal reflection)

            self.z = 2 * self.d - (self.z + s * self.uz)  # mirror
            self.uz = -uz  # change in opposite z-direction (from + to -)

        # elif self.z > self.d: #if photon crosses rear surface (nonreflecting background)
        #
        #     # retract step
        #     self.x -= s * self.ux
        #     self.y -= s * self.uy
        #     self.z -= s * self.uz
        #
        #     #compute new direction and reflection probability
        #     if uz <= self.cosCrit1:   #total internal reflection based on critical angle
        #         r = 1.0
        #     else:
        #         r, uz1 = RFresnel(self.n, self.nr, uz)
        #
        #
        #     if np.random.random_sample() > r:   # moves out of medium through rear surface -> transmission
        #         self.uz = uz1
        #
        #         # ir = int((self.x*self.x + self.y*self.y)**.5/dr)
        #         # T_r[ir] += self.w
        #
        #         self.status = 'Transmitted'
        #         self.alive = False
        #
        #     else:  # move back in the medium (internal reflection)
        #         self.x += s * self.ux
        #         self.y += s * self.uy
        #         self.z = 2*self.d - (self.z + s * self.uz) #mirror
        #
        #         self.uz = -uz  #change in opposite z-direction (from + to -)

    def Drop(self):
        # Drop photon weight inside the sample.
        # The photon is assumed not dead.
        # The weight drop is dw = w*mua/(mua+mus).
        # The dropped weight is assigned to the absorption array
        # elements.

        x = self.x
        y = self.y

        # # compute array indices
        # iz = int(self.z / dz)
        # ir = int((x * x + y * y) ** 0.5 / dr)
        #
        # # check if out of bounds -> move to last 'overflow' bin
        # if iz > (Nz - 1):
        #     iz = Nz - 1
        # if ir > (Nr - 1):
        #     ir = Nr - 1

        # update photon weight.

        dwa = self.w * self.mua / self.mut
        self.w -= dwa

        # A_rz[iz, ir] += dwa         # assign dwa to the absorption array element.

    def Spin(self):
        # Choose a new direction for photon propagation by
        # sampling the polar deflection angle theta and the
        # azimuthal angle psi.
        # Note:
        # theta: 0 - pi so sin(theta) is always positive
        # feel free to use sqrt() for cos(theta).
        # psi:   0 - 2pi
        # for 0-pi  sin(psi) is +
        # for pi-2pi sin(psi) is -

        ux = self.ux
        uy = self.uy
        uz = self.uz

        cost = SpinTheta(self.g)

        # self.Nscattered += 1
        sint = (1.0 - cost * cost) ** 0.5
        # sqrt() is faster than sin().

        psi = 2.0 * np.pi * np.random.random_sample()  # spin psi 0-2pi
        cosp = np.cos(psi)
        if psi < np.pi:
            sinp = (1.0 - cosp * cosp) ** 0.5
            # sqrt() is faster than sin().
        else:
            sinp = -(1.0 - cosp * cosp) ** 0.5

        if np.fabs(uz) > COSZERO:  # normal incident.
            self.ux = sint * cosp
            self.uy = sint * sinp
            self.uz = cost * np.sign(uz)
            # SIGN() is faster than division.
        else:  # regular incident.
            temp = (1.0 - uz * uz) ** 0.5
            self.ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost
            self.uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost
            self.uz = -sint * cosp * temp + uz * cost


    def Roulette(self):
        # The photon weight is small, and the photon packet tries
        # to survive a roulette.
        if self.w == 0.0:
            print('Zeroweight')
            self.status = 'Absorbed'
            self.alive = False
        elif np.random.random_sample() < CHANCE:    # survived the roulette.
            self.w /= CHANCE
        else:
            self.status = 'Absorbed'
            self.alive = False

    def VarnishReflection(self,thickness):  # if photon is reflected, move it back towards the edge of the varnish layer
        r, uz1 = RFresnel(self.n, self.nf, -self.uz)

        self.ux *= self.n / self.nf
        self.uy *= self.n / self.nf
        self.uz = -uz1

        if uz1 < float(COS90D):  # very slant
            # print(self.uz < COS90D)
            # print('slant', self.uz, COS90D)
            ir = int((self.x * self.x + self.y * self.y) ** .5 / self.dr)
            if ir > (self.Nr - 1):
                ir = self.Nr - 1
            self.output = (ir, self.w)
            return
        s = thickness / self.uz
        self.x += s * self.ux
        self.y += s * self.uy
        self.z = thickness

        ir = int((self.x * self.x + self.y * self.y) ** .5 / self.dr)
        # # check if out of bounds -> move to last 'overflow' bin

        if ir > (self.Nr - 1):
            ir = self.Nr - 1
        self.output = (ir, self.w)
        return




class PhotonILT(Photon):
    # A photon that samples cos(theta) using the inverse look-up table method

    def __init__(self, mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr, lookUpTable = 0, lookUpAngles = 0):
        #look-up table sampling of scattered angle
        self.lookUpTable = lookUpTable
        self.lookUpAngles = lookUpAngles
        super().__init__(mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr)

    def Spin(self):
        # Choose a new direction for photon propagation by
        # sampling the polar deflection angle theta and the
        # azimuthal angle psi.
        # Note:
        # theta: 0 - pi so sin(theta) is always positive
        # feel free to use sqrt() for cos(theta).
        # psi:   0 - 2pi
        # for 0-pi  sin(psi) is +
        # for pi-2pi sin(psi) is -

        ux = self.ux
        uy = self.uy
        uz = self.uz

        cost = self.determineAngle()

        # self.Nscattered += 1
        sint = (1.0 - cost * cost) ** 0.5
        # sqrt() is faster than sin().

        psi = 2.0 * np.pi * np.random.random_sample()  # spin psi 0-2pi
        cosp = np.cos(psi)
        if psi < np.pi:
            sinp = (1.0 - cosp * cosp) ** 0.5
            # sqrt() is faster than sin().
        else:
            sinp = -(1.0 - cosp * cosp) ** 0.5

        if np.fabs(uz) > COSZERO:  # clost to normal incident.
            self.ux = sint * cosp
            self.uy = sint * sinp
            self.uz = cost * np.sign(uz)
            # SIGN() is faster than division.
        else:  # regular incident.
            temp = (1.0 - uz * uz) ** 0.5
            self.ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost
            self.uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost
            self.uz = -sint * cosp * temp + uz * cost


    def determineAngle(self):
        # cosine(theta) is sampled using the inverse look-up table method
        zeta = np.random.random_sample()
        #inverse look-up table sampling of scattered angle cos(theta)
        J = int(len(self.lookUpAngles) * zeta)

        try:
            value = (self.lookUpAngles[J + 1] - self.lookUpAngles[J]) / (self.lookUpTable[J + 1] - self.lookUpTable[J]) * (zeta - self.lookUpTable[J]) + \
                    self.lookUpAngles[J]
        except: #if last bin -> J+1 does not exist, isotropic scattering
            value = 1 - np.random.random_sample()*2

        # # Unhash to redistribute very slant angles
        # if value > .99:
        #     value = 1
        # if value < -.99:
        #     value = 1

        return value
