import numpy as np
from scipy import signal as sg


class FluxFootprintPredictor:
    """
    Flux Footprint Prediction (FFP) climatology.
    Based on Kljun et al. (2015), Geosci. Model Dev. 8, 3695-3713.
    https://github.com/Open-ET/flux-data-footprint/tree/master
    """

    # Model constants
    _a, _b, _c, _d = 1.4524, -1.9914, 1.4622, 0.1359
    _ac, _bc, _cc = 2.17, 1.66, 20.0
    _oln = 5000  # neutral Obukhov length limit
    _k = 0.4  # von Kármán constant

    def __init__(
        self,
        zm,
        h,
        ol,
        sigmav,
        ustar,
        wind_dir,
        z0=None,
        umean=None,
        domain=None,
        dx=None,
        dy=None,
        nx=None,
        ny=None,
        rs=None,
        rslayer=0,
        smooth_data=True,
        crop=False,
    ):

        if rs is None:
            rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        self._rslayer = rslayer
        self._smooth = smooth_data
        self._crop = crop

        # Coerce all inputs to lists
        (
            self._zm,
            self._h,
            self._ol,
            self._sigmav,
            self._ustar,
            self._wind_dir,
            self._z0,
            self._umean,
        ) = self._parse_inputs(zm, h, ol, sigmav, ustar, wind_dir, z0, umean)

        self._rs = self._parse_rs(rs)
        self._domain, self._dx, self._dy, self._nx, self._ny = self._parse_domain(
            domain, dx, dy, nx, ny
        )

        # Public results (populated by calculate_ffp())
        self.x_2d = None
        self.y_2d = None
        self.fclim_2d = None
        self.rs = None
        self.fr = None
        self.xr = None
        self.yr = None
        self.n = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_ffp(self):
        """Calculate the footprint climatology and return results as a dict."""
        self._build_grid()
        self._accumulate()
        self._normalise_and_smooth()
        if self._rs is not None:
            self._compute_contours()
        if self._crop:
            self._crop_domain()
        return self.to_dict()

    def to_dict(self):
        """Return results as a plain dictionary."""
        return {
            "x_2d": self.x_2d,
            "y_2d": self.y_2d,
            "fclim_2d": self.fclim_2d,
            "rs": self.rs,
            "fr": self.fr,
            "xr": self.xr,
            "yr": self.yr,
            "n": self.n,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Input parsing
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _listify(v):
        return v if isinstance(v, list) else [v]

    def _parse_inputs(self, zm, h, ol, sigmav, ustar, wind_dir, z0, umean):
        zm, h, ol, sigmav, ustar, wind_dir, z0, umean = map(
            self._listify, [zm, h, ol, sigmav, ustar, wind_dir, z0, umean]
        )

        ts_len = len(ustar)

        if len(zm) == 1:
            zm = [zm[0]] * ts_len

        # Resolve z0 vs umean
        if not all(v is None for v in z0):
            umean = [None] * ts_len
            if len(z0) == 1:
                z0 = [z0[0]] * ts_len
        else:
            z0 = [None] * ts_len

        return zm, h, ol, sigmav, ustar, wind_dir, z0, umean

    @staticmethod
    def _parse_rs(rs):
        import numbers

        if rs is None:
            return None
        if isinstance(rs, numbers.Number):
            rs = [rs]
        rs = [x / 100.0 if x >= 1 else x for x in rs]
        rs = sorted(x for x in rs if x <= 0.9)
        return rs if rs else None

    @staticmethod
    def _parse_domain(domain, dx, dy, nx, ny):
        import numbers

        if isinstance(dx, numbers.Number) and dy is None:
            dy = dx
        if isinstance(dy, numbers.Number) and dx is None:
            dx = dy
        if not all(isinstance(v, numbers.Number) for v in [dx, dy]):
            dx = dy = None
        if isinstance(nx, int) and ny is None:
            ny = nx
        if isinstance(ny, int) and nx is None:
            nx = ny
        if not all(isinstance(v, int) for v in [nx, ny]):
            nx = ny = None
        if not isinstance(domain, list) or len(domain) != 4:
            domain = None

        if all(v is None for v in [dx, nx, domain]):
            domain = [-1000.0, 1000.0, -1000.0, 1000.0]
            dx = dy = 2.0
            nx = ny = 1000
        elif domain is not None:
            if dx is not None:
                nx = int((domain[1] - domain[0]) / dx)
                ny = int((domain[3] - domain[2]) / dy)
            else:
                if nx is None:
                    nx = ny = 1000
                dx = (domain[1] - domain[0]) / float(nx)
                dy = (domain[3] - domain[2]) / float(ny)
        elif dx is not None and nx is not None:
            domain = [-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2]
        elif dx is not None:
            domain = [-1000, 1000, -1000, 1000]
            nx = int((domain[1] - domain[0]) / dx)
            ny = int((domain[3] - domain[2]) / dy)
        elif nx is not None:
            domain = [-1000, 1000, -1000, 1000]
            dx = (domain[1] - domain[0]) / float(nx)
            dy = (domain[3] - domain[2]) / float(nx)

        return domain, dx, dy, nx, ny

    # ──────────────────────────────────────────────────────────────────────────
    # Grid
    # ──────────────────────────────────────────────────────────────────────────

    def _build_grid(self):
        xmin, xmax, ymin, ymax = self._domain
        x = np.linspace(xmin, xmax, self._nx + 1)
        y = np.linspace(ymin, ymax, self._ny + 1)
        self.x_2d, self.y_2d = np.meshgrid(x, y)
        self._rho = np.sqrt(self.x_2d**2 + self.y_2d**2)
        self._theta = np.arctan2(self.x_2d, self.y_2d)
        self.fclim_2d = np.zeros(self.x_2d.shape)

    # ──────────────────────────────────────────────────────────────────────────
    # Validity check
    # ──────────────────────────────────────────────────────────────────────────

    def _is_valid(self, ustar, sigmav, h, ol, wind_dir, zm, z0, umean):
        if any(v is None for v in [ustar, sigmav, h, ol, wind_dir, zm]):
            return False
        if zm <= 0 or h <= 10 or zm > h:
            return False
        if z0 is not None and umean is None:
            if z0 <= 0:
                return False
            if self._rslayer == 0 and zm <= 12.5 * z0:
                return False
        if zm / ol <= -15.5:
            return False
        if sigmav <= 0 or ustar < 0.1:
            return False
        if not (0 <= wind_dir <= 360):
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Core footprint calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _single_footprint(self, ustar, sigmav, h, ol, wind_dir, zm, z0, umean):
        """Return 2-D footprint array for one timestamp, or None if invalid."""
        a, b, c, d = self._a, self._b, self._c, self._d
        ac, bc, cc = self._ac, self._bc, self._cc
        oln, k = self._oln, self._k

        rotated_theta = self._theta - wind_dir * np.pi / 180.0
        f_ci = np.zeros(self.x_2d.shape)
        xstar = np.zeros(self.x_2d.shape)

        if z0 is not None:
            ol_s = ol if ol != 0 else 1e-6
            if ol_s <= 0 or ol_s >= oln:
                xx = (1 - 19.0 * zm / ol_s) ** 0.25
                psi_f = (
                    np.log((1 + xx**2) / 2.0)
                    + 2.0 * np.log((1 + xx) / 2.0)
                    - 2.0 * np.arctan(xx)
                    + np.pi / 2
                )
            else:
                psi_f = -5.3 * zm / ol_s

            denom = np.log(zm / z0) - psi_f
            if denom <= 0:
                return None

            xstar = self._rho * np.cos(rotated_theta) / zm * (1 - zm / h) / denom
            px = np.where(xstar > d)
            fstar = np.zeros(self.x_2d.shape)
            fstar[px] = a * (xstar[px] - d) ** b * np.exp(-c / (xstar[px] - d))
            f_ci[px] = fstar[px] / zm * (1 - zm / h) / denom
        else:
            xstar = (
                self._rho
                * np.cos(rotated_theta)
                / zm
                * (1 - zm / h)
                / (umean / ustar * k)
            )
            px = np.where(xstar > d)
            fstar = np.zeros(self.x_2d.shape)
            fstar[px] = a * (xstar[px] - d) ** b * np.exp(-c / (xstar[px] - d))
            f_ci[px] = fstar[px] / zm * (1 - zm / h) / (umean / ustar * k)

        # Crosswind dispersion σy
        sigystar = np.zeros(self.x_2d.shape)
        sigystar[px] = ac * np.sqrt(bc * xstar[px] ** 2 / (1 + cc * np.abs(xstar[px])))

        ol_eff = -1e6 if abs(ol) > oln else ol
        scale = min(
            1.0, 1e-5 * abs(zm / ol_eff) ** (-1) + (0.80 if ol_eff <= 0 else 0.55)
        )

        sigy = np.zeros(self.x_2d.shape)
        sigy[px] = sigystar[px] / scale * zm * sigmav / ustar
        sigy[sigy < 0] = np.nan

        f_2d = np.zeros(self.x_2d.shape)
        f_2d[px] = (
            f_ci[px]
            / (np.sqrt(2 * np.pi) * sigy[px])
            * np.exp(
                -((self._rho[px] * np.sin(rotated_theta[px])) ** 2)
                / (2 * sigy[px] ** 2)
            )
        )
        return f_2d

    def _accumulate(self):
        valids = []
        for args in zip(
            self._ustar,
            self._sigmav,
            self._h,
            self._ol,
            self._wind_dir,
            self._zm,
            self._z0,
            self._umean,
        ):
            ustar, sigmav, h, ol, wd, zm, z0, umean = args
            if not self._is_valid(ustar, sigmav, h, ol, wd, zm, z0, umean):
                valids.append(False)
                continue
            f_2d = self._single_footprint(ustar, sigmav, h, ol, wd, zm, z0, umean)
            if f_2d is None:
                valids.append(False)
                continue
            self.fclim_2d += f_2d
            valids.append(True)
        self.n = sum(valids)

    # ──────────────────────────────────────────────────────────────────────────
    # Post-processing
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_and_smooth(self):
        if self.n == 0:
            return
        self.fclim_2d /= self.n
        if self._smooth:
            kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
            self.fclim_2d = sg.convolve2d(self.fclim_2d, kernel, mode="same")
            self.fclim_2d = sg.convolve2d(self.fclim_2d, kernel, mode="same")

    def _contour_levels(self, rs):
        """Return list of (r, cumulative_area, f_level) for each r in rs."""
        from numpy import ma

        dx, dy = self._dx, self._dy
        f = self.fclim_2d
        sf = np.sort(f, axis=None)[::-1]
        msf = ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))
        csf = msf.cumsum().filled(np.nan) * dx * dy
        levels = []
        for r in rs:
            idx = np.nanargmin(np.abs(csf - r))
            levels.append((round(r, 3), float(csf[idx]), float(sf[idx])))
        return levels

    def _contour_vertices(self, level):
        """Extract ordered (xr, yr) contour polygon at given f-level."""
        try:
            from skimage import measure

            contours = measure.find_contours(self.fclim_2d, level)
            if not contours:
                return None, None
            seg = max(contours, key=len)
            ny_g, nx_g = self.fclim_2d.shape
            xr = np.interp(seg[:, 1], np.arange(nx_g), self.x_2d[0, :]).tolist()
            yr = np.interp(seg[:, 0], np.arange(ny_g), self.y_2d[:, 0]).tolist()
        except ImportError:
            # Pure-numpy fallback: collect edge-straddling midpoints
            above = self.fclim_2d >= level
            pts_x, pts_y = [], []
            rows, cols = np.where(above[:-1, :] ^ above[1:, :])
            for r, c in zip(rows, cols):
                t = (level - self.fclim_2d[r, c]) / (
                    self.fclim_2d[r + 1, c] - self.fclim_2d[r, c] + 1e-30
                )
                pts_x.append(float(self.x_2d[r, c]))
                pts_y.append(
                    float(self.y_2d[r, c]) + t * (self.y_2d[r + 1, c] - self.y_2d[r, c])
                )
            rows, cols = np.where(above[:, :-1] ^ above[:, 1:])
            for r, c in zip(rows, cols):
                t = (level - self.fclim_2d[r, c]) / (
                    self.fclim_2d[r, c + 1] - self.fclim_2d[r, c] + 1e-30
                )
                pts_x.append(
                    float(self.x_2d[r, c]) + t * (self.x_2d[r, c + 1] - self.x_2d[r, c])
                )
                pts_y.append(float(self.y_2d[r, c]))
            if not pts_x:
                return None, None
            cx, cy = np.mean(pts_x), np.mean(pts_y)
            order = np.argsort(np.arctan2(np.array(pts_y) - cy, np.array(pts_x) - cx))
            xr = [pts_x[i] for i in order]
            yr = [pts_y[i] for i in order]

        # Invalidate if contour touches the domain boundary
        if (
            self.x_2d.min() >= min(xr)
            or max(xr) >= self.x_2d.max()
            or self.y_2d.min() >= min(yr)
            or max(yr) >= self.y_2d.max()
        ):
            return None, None
        return xr, yr

    def _compute_contours(self):
        clevs = self._contour_levels(self._rs)
        self.rs = [lv[0] for lv in clevs]
        self.fr = [lv[2] for lv in clevs]
        self.xr, self.yr = [], []
        for lv in clevs:
            xr, yr = self._contour_vertices(lv[2])
            self.xr.append(xr)
            self.yr.append(yr)

    def _crop_domain(self):
        """Crop grids to the extent of the largest valid contour."""
        if not self.xr:
            return
        # Find the outermost valid contour
        valid_pairs = [(xr, yr) for xr, yr in zip(self.xr, self.yr) if xr is not None]
        if not valid_pairs:
            return
        xr_last, yr_last = valid_pairs[-1]
        dminx = np.floor(min(xr_last))
        dmaxx = np.ceil(max(xr_last))
        dminy = np.floor(min(yr_last))
        dmaxy = np.ceil(max(yr_last))

        def _idx_range(arr_1d, lo, hi, max_idx):
            idx = np.where((arr_1d >= lo) & (arr_1d <= hi))[0]
            if len(idx) == 0:
                return np.arange(max_idx)
            return np.clip(
                np.concatenate([[idx[0] - 1], idx, [idx[-1] + 1]]), 0, max_idx - 1
            )

        irange = _idx_range(self.x_2d[0, :], dminx, dmaxx, self.x_2d.shape[1])
        jrange = _idx_range(self.y_2d[:, 0], dminy, dmaxy, self.y_2d.shape[0])
        jrange = jrange[:, None]

        self.x_2d = self.x_2d[jrange, irange]
        self.y_2d = self.y_2d[jrange, irange]
        self.fclim_2d = self.fclim_2d[jrange, irange]


def get_utm_zone_from_longlat(longitude, latitude):
    """
    Determines the UTM zone number and hemisphere (N/S) for a given latitude and longitude.

    Parameters:
    -----------
    longitude (x): float
        Longitude coordinate in decimal degrees.
    latitude (y): float
        Latitude coordinate in decimal degrees.

    Returns:
    --------
    str
        UTM zone with hemisphere (e.g., "34N", "35N") and EPSG codes.

    Example:
    --------
    >>> get_utm_zone(31.2357, 30.0444)  # Cairo, Egypt
    '36N'
    """
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be within the range [-180, 180].")
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be within the range [-90, 90].")
    # Calculate UTM zone number
    utm_zone = int((longitude + 180) / 6) + 1

    # Determine hemisphere
    hemisphere = "N" if latitude >= 0 else "S"
    zone_code = f"{utm_zone}{hemisphere}"
    zone = zone_code[:-1]
    epsg_code = f"EPSG:326{zone}" if hemisphere == "N" else f"EPSG:327{zone}"
    return {"zone": zone_code, "epsg": epsg_code}


def create_polygons_from_footprint(fp, x_tower, y_tower):
    """Create polygons from footprint data and shift them to the tower location.
    Args:
        fp: dict containing footprint data with keys 'rs', 'xr', 'yr'
        x_tower: x coordinate of the tower location in longitude (geographic coordinates)
        y_tower: y coordinate of the tower location in latitude (geographic coordinates)
    Returns:
        GeoDataFrame containing the footprint polygons with a 'level' attribute
    """
    import geopandas as gpd
    from shapely.affinity import translate
    from shapely.geometry import Point, Polygon

    # create point from tower location
    crs = get_utm_zone_from_longlat(x_tower, y_tower)["epsg"]
    point = gpd.GeoDataFrame(
        [{"geometry": Point(x_tower, y_tower), "label": "tower"}], crs="EPSG:4326"
    ).to_crs(crs)
    x = point.geometry.x.values[0]
    y = point.geometry.y.values[0]
    polygons = []
    for i, r in enumerate(fp["rs"]):
        xr = fp["xr"][i]
        yr = fp["yr"][i]
        if xr is not None and yr is not None:
            # Crete the first geopandas GeoDataFrame with the footprint polygons
            first = gpd.GeoDataFrame(
                {"level": [r], "geometry": [Polygon(zip(fp["xr"][0], fp["yr"][0]))]},
                crs=crs,
            )
            x_center = first.geometry[0].centroid.x
            y_center = first.geometry[0].centroid.y
            coords = list(zip(xr, yr))
            poly = Polygon(coords)
            poly = translate(
                poly, xoff=x - x_center, yoff=y - y_center
            )  # shift to tower location
            polygons.append({"level": int(r * 100), "geometry": poly})

    gdf = gpd.GeoDataFrame(polygons, crs=crs)  # set your CRS
    return gdf


def create_point_from_tower_location(x_tower, y_tower):
    """Create a point geometry from the tower location.
    Args:
        x_tower: x coordinate of the tower location in longitude (geographic coordinates)
        y_tower: y coordinate of the tower location in latitude (geographic coordinates)
    Returns:
        GeoDataFrame containing the tower point geometry with a 'label' attribute
    """
    import geopandas as gpd
    from shapely.geometry import Point

    point = gpd.GeoDataFrame(
        [{"geometry": Point(x_tower, y_tower), "label": "tower"}], crs="EPSG:4326"
    )
    return point
